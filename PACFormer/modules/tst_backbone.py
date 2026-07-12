__all__ = ["PatchTST_backbone"]

from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .patchtst_layers import Transpose, positional_encoding
import math
import csv


class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        num_groups: int,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(
            num_groups,
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
                d_model=d_model,
            )

        self.res_attention = res_attention

    def forward(self, z):
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [B, N, P, patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [B, N, patch_len, P]

        if self.res_attention:
            z, attns, alphas, _ = self.backbone(z)
        else:
            z = self.backbone(z)

        z = self.head(z)  # z: [bs x nvars x target_window]

        if self.res_attention:
            return z, attns, alphas
        else:
            return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(nn.Module):
    def __init__(
        self, individual, n_vars, nf, target_window, head_dropout=0, d_model=512
    ):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, d_model)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
        self,
        num_groups,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        self.res_attention = res_attention

        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            num_groups,
            c_in,
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
            stations_csv_path=kwargs.get("stations_csv_path", None),
            use_axial_rope=kwargs.get("use_axial_rope", True),
            rope_base_time=kwargs.get("rope_base_time", 10000),
            rope_base_space=kwargs.get("rope_base_space", 10000),
            q_event_mode=kwargs.get("q_event_mode", "event"),
        )

    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)  # x: [B, N, P, patch_len]
        x = self.W_P(x)  # x: [B, N, P, E]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # B*N, P, E
        u = self.dropout(u + self.W_pos)

        if self.res_attention:
            z, attns, alphas = self.encoder(u)  # z: [B*N, P, E]
        else:
            z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]

        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        if self.res_attention:
            return z, attns, alphas, self.W_pos
        else:
            return z


class TSTEncoder(nn.Module):
    def __init__(
        self,
        num_groups,
        c_in,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
        stations_csv_path: Optional[str] = None,
        use_axial_rope: bool = True,
        rope_base_time: int = 10000,
        rope_base_space: int = 10000,
        q_event_mode: str = "event",
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    num_groups,
                    c_in,
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                    stations_csv_path=stations_csv_path,
                    use_axial_rope=use_axial_rope,
                    rope_base_time=rope_base_time,
                    rope_base_space=rope_base_space,
                    q_event_mode=q_event_mode,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        output = src
        attn = None
        attns = []
        alphas = []

        if self.res_attention:
            for mod in self.layers:
                output, attn, alpha = mod(
                    output,
                    prev=attn,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
                attns.append(attn)
                alphas.append(alpha)
            return output, attns, alphas
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_groups,
        c_in,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        stations_csv_path: Optional[str] = None,
        use_axial_rope: bool = True,
        rope_base_time: int = 10000,
        rope_base_space: int = 10000,
        q_event_mode: str = "event",
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super().__init__()
        assert not d_model % n_heads, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            num_groups,
            c_in,
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
            stations_csv_path=stations_csv_path,
            use_axial_rope=use_axial_rope,
            rope_base_time=rope_base_time,
            rope_base_space=rope_base_space,
            q_event_mode=q_event_mode,
        )

        if "batch" in norm.lower():
            self.pre_norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.pre_norm_attn = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src = self.pre_norm_attn(src)

        if self.res_attention:
            src2, attn, alpha = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2 = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn

        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        src2 = self.ff(src)

        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, attn, alpha
        else:
            return src


def select_repr_groups(scores, A, k=3, lam=0.3):
    """Select k groups by membership sharpness and sequential diversity."""
    B, N, _ = scores.shape
    eps = 1e-9

    H = -(A * (A + eps).log()).sum(dim=1)  # [B,N] (column entropy)
    sharp = -H  # higher better (sharpness member)

    idx = torch.zeros(B, k, dtype=torch.long, device=scores.device)

    idx[:, 0] = sharp.argmax(dim=1)

    sel0 = idx[:, 0].view(B, 1, 1).expand(B, 1, N)
    sim_to_sel = scores.gather(1, sel0).squeeze(1)

    for t in range(1, k):
        diversity = -sim_to_sel
        score_t = lam * sharp + (1 - lam) * diversity

        score_t.scatter_(1, idx[:, :t], float("-inf"))

        idx[:, t] = score_t.argmax(dim=1)

        sel = idx[:, t].view(B, 1, 1).expand(B, 1, N)
        new_sim = scores.gather(1, sel).squeeze(1)  # [B,N]
        sim_to_sel = torch.maximum(sim_to_sel, new_sim)

    return idx


def causal_mask_pp(P: int, device, dtype=torch.bool):
    return torch.triu(torch.ones(P, P, device=device, dtype=dtype), diagonal=1)


def causal_mask_kP(k: int, P: int, device, dtype=torch.bool):
    p = torch.arange(P, device=device).repeat(k)  # [kP] group-major order
    return (p[None, :] > p[:, None]).to(dtype)  # [kP, kP]


def _haversine_pairwise_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distance matrix (km). lat/lon in degrees, shape [N]."""
    R = 6371.0088  # mean Earth radius in km
    lat = np.deg2rad(lat_deg).reshape(-1, 1)
    lon = np.deg2rad(lon_deg).reshape(-1, 1)

    dlat = lat - lat.T
    dlon = lon - lon.T

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a + 1e-12))
    return (R * c).astype(np.float64)


def _mds_1d_from_dist(D: np.ndarray) -> np.ndarray:
    """Classical MDS 1D embedding from distance matrix D [N,N]."""
    N = D.shape[0]
    D2 = (D**2).astype(np.float64)
    J = np.eye(N) - np.ones((N, N), dtype=np.float64) / N
    B = -0.5 * (J @ D2 @ J)

    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    if w[0] <= 1e-12:
        return np.zeros((N,), dtype=np.float64)

    x = np.sqrt(w[0]) * v[:, 0]
    x = x - x.mean()
    x = x / (x.std() + 1e-12)
    return x.astype(np.float64)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1,x2)->(-x2,x1) for RoPE, last-dim must be even."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_1d(
    x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """Apply standard 1D RoPE to x using positions pos.
    x: [..., D] where D even
    pos: broadcastable to x.shape[:-1] (e.g., [B,L] or [1,L])
    inv_freq: [D/2]
    """
    theta = pos.unsqueeze(-1) * inv_freq
    cos = torch.cos(theta).repeat_interleave(2, dim=-1)
    sin = torch.sin(theta).repeat_interleave(2, dim=-1)
    return (x * cos) + (_rotate_half(x) * sin)


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        num_groups,
        c_in,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=True,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
        stations_csv_path: Optional[str] = None,
        use_axial_rope: bool = True,
        rope_base_time: int = 10000,
        rope_base_space: int = 10000,
        q_event_mode: str = "event",
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """

        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

        self.n_vars = c_in
        self.G = num_groups

        self.pre_norm_attn = nn.LayerNorm(d_model)

        self.use_axial_rope = bool(use_axial_rope) and (stations_csv_path is not None)
        self.stations_csv_path = stations_csv_path

        time_rope_dim = d_model // 2
        time_rope_dim = time_rope_dim - (time_rope_dim % 2)
        space_rope_dim = d_model - time_rope_dim
        space_rope_dim = space_rope_dim - (space_rope_dim % 2)
        rest_dim = d_model - (time_rope_dim + space_rope_dim)

        self.rope_time_dim = int(time_rope_dim)
        self.rope_space_dim = int(space_rope_dim)
        self.rope_rest_dim = int(rest_dim)

        if self.rope_time_dim > 0:
            inv_t = 1.0 / (
                rope_base_time
                ** (torch.arange(0, self.rope_time_dim, 2).float() / self.rope_time_dim)
            )
            self.register_buffer("rope_inv_freq_time", inv_t, persistent=False)
        else:
            self.register_buffer("rope_inv_freq_time", torch.empty(0), persistent=False)

        if self.rope_space_dim > 0:
            inv_s = 1.0 / (
                rope_base_space
                ** (
                    torch.arange(0, self.rope_space_dim, 2).float()
                    / self.rope_space_dim
                )
            )
            self.register_buffer("rope_inv_freq_space", inv_s, persistent=False)
        else:
            self.register_buffer(
                "rope_inv_freq_space", torch.empty(0), persistent=False
            )

        self.register_buffer(
            "node_pos1d", torch.zeros(c_in, dtype=torch.float32), persistent=False
        )

        self.use_qbar_space_rope = True
        self.rope_base_space_qbar = float(rope_base_space)
        self.register_buffer(
            "rope_inv_freq_qbar_space", torch.empty(0), persistent=False
        )

        if self.use_axial_rope:
            with open(stations_csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                cols_l = [c.lower() for c in cols]

                def _pick(names):
                    for n in names:
                        if n in cols_l:
                            return cols[cols_l.index(n)]
                    return None

                lat_col = _pick(["lat", "latitude", "y", "y_lat"])
                lon_col = _pick(["lon", "longitude", "lng", "x", "x_lon"])
                if lat_col is None or lon_col is None:
                    raise ValueError(
                        f"Cannot find lat/lon columns in stations_csv_path. Columns={cols}"
                    )

                lat_list, lon_list = [], []
                for row in reader:
                    lat_list.append(float(row[lat_col]))
                    lon_list.append(float(row[lon_col]))

            if len(lat_list) != c_in:
                raise ValueError(
                    f"stations_csv_path node count ({len(lat_list)}) != c_in ({c_in}). "
                    f"Make sure the CSV rows align with BNPE node order."
                )

            lat_np = np.asarray(lat_list, dtype=np.float64)
            lon_np = np.asarray(lon_list, dtype=np.float64)

            D = _haversine_pairwise_km(lat_np, lon_np)  # [N,N] km
            pos1d = _mds_1d_from_dist(D)  # [N]
            self.node_pos1d.copy_(torch.tensor(pos1d, dtype=torch.float32))

        self.use_leadlag = True
        self.leadlag_min = (
            1  # enforce true lag (key time < query time). set 0 to allow same-time.
        )
        self.leadlag_max = 4  # max lag in patch steps (tune; e.g., 6*patch_len hours)
        if q_event_mode not in {"event", "original", "add"}:
            raise ValueError("q_event_mode must be one of {'event', 'original', 'add'}")
        self.q_event_mode = q_event_mode
        if q_event_mode in {"original", "add"}:
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        if q_event_mode in {"event", "add"}:
            self.W_Qe = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
            self.q_event_norm = nn.LayerNorm(d_model)
        if q_event_mode == "add":
            self.q_event_alpha = nn.Parameter(torch.tensor(0.5))

    def _apply_axial_rope_to_tok(
        self, tok: torch.Tensor, idx: torch.Tensor, P: int
    ) -> torch.Tensor:
        """Axial RoPE on W_rep tokens (g,p) in group-major order.
        tok: [B, kP, E]
        idx: [B, k]  (selected representative group ids in [0, N))
        """
        if (not self.use_axial_rope) or (
            (self.rope_time_dim + self.rope_space_dim) == 0
        ):
            return tok

        B, L, E = tok.shape
        k = idx.size(1)
        device = tok.device
        dtype = tok.dtype

        p_idx = torch.arange(P, device=device).repeat(k)  # [kP]
        pos_t = p_idx[None, :].to(dtype)  # [1, kP] (broadcast over batch)

        pos_s_g = self.node_pos1d.to(device=device, dtype=dtype)[idx]  # [B, k]
        pos_s = pos_s_g.repeat_interleave(P, dim=1)  # [B, kP]

        t0 = 0
        t1 = self.rope_time_dim
        s1 = t1 + self.rope_space_dim

        x_t = tok[..., t0:t1]
        x_s = tok[..., t1:s1]

        if self.rope_time_dim > 0:
            x_t = _apply_rope_1d(
                x_t, pos_t, self.rope_inv_freq_time.to(device=device, dtype=dtype)
            )
        if self.rope_space_dim > 0:
            x_s = _apply_rope_1d(
                x_s, pos_s, self.rope_inv_freq_space.to(device=device, dtype=dtype)
            )

        return torch.cat([x_t, x_s], dim=-1)

    def _apply_spatial_rope_to_descriptor(self, descriptor):
        if not self.use_qbar_space_rope or self.stations_csv_path is None:
            return descriptor

        feature_dim = descriptor.size(-1)
        rope_dim = feature_dim - feature_dim % 2
        if rope_dim < 2:
            return descriptor
        if self.rope_inv_freq_qbar_space.numel() != rope_dim // 2:
            inv_freq = 1.0 / (
                self.rope_base_space_qbar
                ** (torch.arange(0, rope_dim, 2).float() / rope_dim)
            )
            self._buffers["rope_inv_freq_qbar_space"] = inv_freq

        inv_freq = self.rope_inv_freq_qbar_space.to(
            device=descriptor.device, dtype=descriptor.dtype
        )
        position = self.node_pos1d.to(device=descriptor.device, dtype=descriptor.dtype)[
            None
        ].expand(descriptor.size(0), -1)
        rotated = _apply_rope_1d(descriptor[..., :rope_dim], position, inv_freq)
        return torch.cat([rotated, descriptor[..., rope_dim:]], dim=-1)

    def dynamic_group_construction(self, query, key):
        """Construct sparse candidate groups and select representative groups."""
        batch_size, num_sites, num_patches, d_model = query.shape
        descriptor = F.normalize(query.reshape(batch_size, num_sites, -1), dim=-1)
        descriptor = self._apply_spatial_rope_to_descriptor(descriptor)
        similarity = torch.einsum("bnd,bmd->bnm", descriptor, descriptor)
        similarity = similarity / descriptor.size(-1) ** 0.5

        membership = torch.softmax(similarity / 0.5, dim=1)
        top_sites = min(int(math.sqrt(self.G)), membership.size(1))
        values, indices = torch.topk(membership, k=top_sites, dim=1)
        sparse_membership = torch.zeros_like(membership)
        sparse_membership.scatter_(1, indices, values)
        membership = sparse_membership / sparse_membership.sum(dim=1, keepdim=True)

        candidate_groups = torch.einsum("bng,bnpe->bgpe", membership, key)
        representative_indices = select_repr_groups(
            similarity, membership, k=self.G, lam=0.1
        )
        gather_index = representative_indices[:, :, None, None].expand(
            batch_size, self.G, num_patches, d_model
        )
        representative_groups = torch.gather(
            candidate_groups, dim=1, index=gather_index
        )
        return (
            similarity,
            candidate_groups,
            representative_indices,
            representative_groups,
        )

    def cross_group_propagation(self, representative_groups, representative_indices):
        """Model causal, lead-lag, down-emphasized interactions across groups."""
        batch_size, _, num_patches, d_model = representative_groups.shape
        common_mode = representative_groups.mean(dim=1, keepdim=True)
        common_direction = F.normalize(common_mode, dim=-1, eps=1e-6)
        centered = representative_groups - common_mode
        down_score = F.relu(-(centered * common_direction).sum(dim=-1))
        down_mask = down_score > 0

        tokens = centered.reshape(batch_size, self.G * num_patches, d_model)
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        tokens = self._apply_axial_rope_to_tok(
            tokens, representative_indices, num_patches
        )
        score = torch.matmul(tokens, tokens.transpose(-1, -2))

        causal = causal_mask_kP(
            self.G, num_patches, device=score.device, dtype=torch.bool
        )
        score = score.masked_fill(causal[None], float("-inf"))
        if self.use_leadlag:
            patch_index = torch.arange(num_patches, device=score.device).repeat(self.G)
            lag = patch_index[:, None] - patch_index[None, :]
            valid_lag = (lag >= self.leadlag_min) & (lag <= self.leadlag_max)
            score = score.masked_fill(~valid_lag[None], float("-inf"))

        down_flat = down_mask.reshape(batch_size, self.G * num_patches)
        valid_transition = down_flat[:, :, None] & down_flat[:, None, :]
        score = score.masked_fill(~valid_transition, float("-inf"))

        valid_score = torch.isfinite(score)
        score = torch.relu(score).masked_fill(~valid_score, float("-inf"))
        top_links = min(int(self.G * 1.5), score.size(-1))
        values, indices = score.topk(top_links, dim=-1)
        sparse_score = torch.full_like(score, float("-inf"))
        sparse_score.scatter_(-1, indices, values)

        invalid_rows = ~torch.isfinite(sparse_score).any(dim=-1, keepdim=True)
        if invalid_rows.any():
            diagonal = torch.eye(
                sparse_score.size(-1), device=score.device, dtype=torch.bool
            )[None]
            sparse_score = sparse_score.masked_fill(invalid_rows & diagonal, 0.0)

        propagation_weight = torch.softmax(sparse_score / 0.2, dim=-1)
        flat_groups = representative_groups.reshape(
            batch_size, self.G * num_patches, d_model
        )
        propagated_groups = torch.matmul(propagation_weight, flat_groups)
        propagated_groups = propagated_groups.view(
            batch_size, self.G, num_patches, d_model
        )
        return propagated_groups, propagation_weight

    def context_diffusion(
        self, similarity, representative_indices, propagated_groups, candidate_groups
    ):
        """Diffuse representative propagation context back to all PV sites."""
        batch_size, num_sites, _ = similarity.shape
        affinity = similarity.gather(
            2, representative_indices[:, None].expand(batch_size, num_sites, self.G)
        )
        top_groups = min(int(math.sqrt(self.G)), self.G)
        values, indices = torch.topk(affinity, k=top_groups, dim=-1)
        sparse_affinity = torch.full_like(affinity, float("-inf"))
        sparse_affinity.scatter_(-1, indices, values)
        diffusion_weight = torch.softmax(sparse_affinity / 0.1, dim=-1)
        site_context = torch.einsum(
            "bnk,bkpe->bnpe", diffusion_weight, propagated_groups
        )
        memory = self.pre_norm_attn(0.2 * candidate_groups + 0.8 * site_context)
        return site_context, memory

    def _project_query(self, query, site_context):
        batch_size, num_sites, num_patches, _ = query.shape
        if self.q_event_mode == "original":
            return (
                self.W_Q(query)
                .reshape(batch_size, num_sites, num_patches, self.n_heads, self.d_k)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )

        event = self.q_event_norm(site_context - query)
        event_query = (
            self.W_Qe(event)
            .reshape(batch_size, num_sites, num_patches, self.n_heads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        if self.q_event_mode == "event":
            return event_query
        base_query = (
            self.W_Q(query)
            .reshape(batch_size, num_sites, num_patches, self.n_heads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        return base_query + self.q_event_alpha * event_query

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        del V, attn_mask
        batch_size = Q.size(0) // self.n_vars
        num_sites = self.n_vars
        num_patches = Q.size(1)
        d_model = Q.size(-1)
        key = Q if K is None else K
        query = Q.reshape(batch_size, num_sites, num_patches, d_model)
        key = key.reshape(batch_size, num_sites, num_patches, d_model)

        similarity, candidate_groups, representative_indices, representative_groups = (
            self.dynamic_group_construction(query, key)
        )
        propagated_groups, propagation_weight = self.cross_group_propagation(
            representative_groups, representative_indices
        )
        site_context, memory = self.context_diffusion(
            similarity, representative_indices, propagated_groups, candidate_groups
        )

        projected_key = (
            self.W_K(memory)
            .reshape(batch_size, num_sites, num_patches, self.n_heads, self.d_k)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
        )
        projected_value = (
            self.W_V(memory)
            .reshape(batch_size, num_sites, num_patches, self.n_heads, self.d_v)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        projected_query = self._project_query(query, site_context)

        causal = causal_mask_pp(num_patches, device=query.device, dtype=torch.bool)[
            None
        ]
        if self.res_attention:
            output, attention_weight = self.sdp_attn(
                projected_query,
                projected_key,
                projected_value,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=causal,
            )
        else:
            output = self.sdp_attn(
                projected_query,
                projected_key,
                projected_value,
                key_padding_mask=key_padding_mask,
                attn_mask=causal,
            )

        output = output.permute(0, 2, 3, 1, 4).contiguous()
        output = output.view(
            batch_size * num_sites,
            num_patches,
            self.n_heads * self.d_v,
        )
        output = self.to_out(output)
        if self.res_attention:
            return output, attention_weight, propagation_weight
        return output


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        attn_scores = (
            torch.matmul(q, k) * self.scale
        )  # attn_scores : [bs x n_heads x max_q_len x q_len]

        if prev is not None:
            attn_scores = attn_scores + prev

        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if (
            key_padding_mask is not None
        ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(
            attn_weights, v
        )  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights
        else:
            return output
