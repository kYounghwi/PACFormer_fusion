import torch
import torch.nn as nn

from .attention import AttentionLayer, FullAttention
from .embedding import InvertedDataEmbedding
from .nwp_branch import NWPBranchPanguTime3D
from .transformer import Decoder, DecoderLayer, Encoder, EncoderLayer
from .tst_backbone import PatchTST_backbone


class Model(nn.Module):
    """PACFormer with asymmetric PV-NWP fusion."""

    def __init__(self, configs, **kwargs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.node_num = configs.c_out

        self.enc_embedding = InvertedDataEmbedding(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        # Global-context branch
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )

        # Group-propagation branch
        self.TST = PatchTST_backbone(
            num_groups=configs.num_groups,
            c_in=configs.enc_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            max_seq_len=1024,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_k=None,
            d_v=None,
            d_ff=configs.d_ff,
            norm=configs.norm,
            attn_dropout=0.0,
            dropout=configs.dropout,
            act="gelu",
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe="zeros",
            learn_pe=True,
            fc_dropout=0.05,
            head_dropout=0.0,
            padding_patch=configs.padding_patch,
            pretrain_head=False,
            head_type="flatten",
            individual=False,
            verbose=False,
            stations_csv_path=configs.stations_csv_path,
            q_event_mode=getattr(configs, "q_event_mode", "event"),
            **kwargs,
        )

        self.decoder = self._build_decoder(configs)
        self.decoder2 = self._build_decoder(configs)
        self.post_spatial = self._alignment(configs.d_model)
        self.post_temporal = self._alignment(configs.d_model)
        self.post_nwp = self._alignment(configs.d_model)
        self.post_fused = self._alignment(configs.d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model), nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model), nn.Sigmoid()
        )

        self.nwp_branch = NWPBranchPanguTime3D(
            time_len=configs.time_len,
            n_vars=configs.nwp_n_vars,
            embed_dim=configs.d_model,
            grid_w=configs.spatial_resolution[1],
            grid_h=configs.spatial_resolution[0],
            num_sensors=configs.enc_in,
            num_heads=configs.n_heads,
            patch_size=configs.cube_patch,
            window_size=(2, 6, 12),
            depth=configs.nwp_vit_layers,
            use_stats=getattr(configs, "nwp_use_stats", False),
            stats_mean=getattr(configs, "nwp_mean", None),
            stats_std=getattr(configs, "nwp_std", None),
            pooling=configs.pooling,
        )
        self.out_nwp = nn.Linear(configs.d_model, configs.pred_len)
        self.out_pv_corr = nn.Linear(configs.d_model, configs.pred_len)
        alpha = torch.linspace(1.0, 0.2, configs.pred_len).view(1, configs.pred_len, 1)
        self.register_buffer("alpha", alpha)

    @staticmethod
    def _alignment(d_model):
        return nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    @staticmethod
    def _build_decoder(configs):
        return Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

    def _encode_pv(self, x_enc, x_mark_enc):
        num_sites = x_enc.size(-1)
        global_tokens = self.enc_embedding(x_enc, x_mark_enc)
        global_context, _ = self.encoder(global_tokens, attn_mask=None)
        global_context = self.post_spatial(global_context[:, :num_sites])

        propagation_output = self.TST(x_enc.permute(0, 2, 1).contiguous())
        propagation_context = (
            propagation_output[0]
            if isinstance(propagation_output, tuple)
            else propagation_output
        )
        propagation_context = self.post_temporal(propagation_context)

        joint_context = torch.cat([global_context, propagation_context], dim=1)
        refined = self.decoder(joint_context, joint_context)
        refined_global = refined[:, :num_sites]
        refined_propagation = refined[:, num_sites:]
        gate = self.gate(torch.cat([refined_global, refined_propagation], dim=-1))
        return gate * refined_global + (1.0 - gate) * refined_propagation

    def _asymmetric_pv_nwp_fusion(self, pv_context, nwp_field):
        num_sites = pv_context.size(1)
        nwp_context = self.post_nwp(self.nwp_branch(nwp_field))
        pv_context = self.post_fused(pv_context)

        nwp_base = self.out_nwp(nwp_context).permute(0, 2, 1)
        joint_context = torch.cat([nwp_context, pv_context], dim=1)
        refined = self.decoder2(joint_context, joint_context)
        refined_nwp = refined[:, :num_sites]
        refined_pv = refined[:, num_sites:]

        gate = self.gate2(torch.cat([refined_nwp, refined_pv], dim=-1))
        correction_token = gate * (refined_pv - refined_nwp)
        pv_correction = self.out_pv_corr(correction_token).permute(0, 2, 1)
        return nwp_base + self.alpha * pv_correction

    def forecast(self, x_enc, x_mark_enc, nwp_y):
        pv_context = self._encode_pv(x_enc, x_mark_enc)
        return self._asymmetric_pv_nwp_fusion(pv_context, nwp_y)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nwp_y, mask=None):
        del x_dec, x_mark_dec, mask
        return self.forecast(x_enc, x_mark_enc, nwp_y)[:, -self.pred_len :]
