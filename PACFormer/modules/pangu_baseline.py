import torch.nn as nn

from .nwp_branch import NWPBranchPanguTime3D


class PanguBaseline(nn.Module):
    """NWP-only Pangu-style baseline used in the original experiments."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
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
        self.output_projection = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, nwp_y, mask=None):
        del x_enc, x_mark_enc, x_dec, x_mark_dec, mask
        nwp_tokens = self.nwp_branch(nwp_y)
        return self.output_projection(nwp_tokens).transpose(1, 2)
