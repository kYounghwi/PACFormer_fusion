import torch
import torch.nn as nn
import torch.nn.functional as F


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd = torch.floor(rnd)
        return x / keep_prob * rnd


def window_partition_3d(x, window_size):
    """
    x: (B, T, H, W, C)
    return windows: (B*nW, wt*wh*ww, C)
    """
    B, T, H, W, C = x.shape
    wt, wh, ww = window_size
    x = x.view(B, T // wt, wt, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1, wt * wh * ww, C)
    return windows


def window_reverse_3d(windows, window_size, B, T, H, W, C):
    """
    windows: (B*nW, wt*wh*ww, C)
    return x: (B, T, H, W, C)
    """
    wt, wh, ww = window_size
    nT = T // wt
    nH = H // wh
    nW = W // ww
    x = windows.view(B, nT, nH, nW, wt, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, T, H, W, C)
    return x


def compute_shift_mask_3d(T, H, W, window_size, shift_size, device):
    """
    Build the shifted-window attention mask used by 3D Swin blocks.
    return: (nW, N, N)
    """
    wt, wh, ww = window_size
    st, sh, sw = shift_size

    img_mask = torch.zeros((1, T, H, W, 1), device=device)  # (1,T,H,W,1)
    cnt = 0

    t_slices = (slice(0, -wt), slice(-wt, -st), slice(-st, None))
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))

    for ts in t_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, ts, hs, ws, :] = cnt
                cnt += 1

    mask_windows = window_partition_3d(img_mask, window_size)  # (nW, N, 1)
    mask_windows = mask_windows.view(-1, wt * wh * ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(
        attn_mask == 0, 0.0
    )
    return attn_mask


class WindowAttention3D(nn.Module):
    """
    3D window self-attention with learnable 3D relative position bias.
    x: (B*nW, N, C), N=wt*wh*ww
    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = tuple(window_size)  # (wt,wh,ww)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim**-0.5

        wt, wh, ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wt - 1) * (2 * wh - 1) * (2 * ww - 1), self.num_heads)
        )

        coords_t = torch.arange(wt)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(
            torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij")
        )  # (3,wt,wh,ww)
        coords_flatten = torch.flatten(coords, 1)  # (3, N)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (3,N,N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N,N,3)

        relative_coords[:, :, 0] += wt - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1

        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= 2 * ww - 1
        relative_position_index = relative_coords.sum(-1)  # (N,N)
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        x: (B*nW, N, C)
        mask: (nW, N, N) or None
        """
        BnW, N, C = x.shape

        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BnW, nH, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (BnW, nH, N, N)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()  # (nH,N,N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BnW // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinBlock3D(nn.Module):
    """
    Apply 3D window attention with an optional cyclic shift.
    x: (B, T, H, W, C)
    """

    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.num_heads = int(num_heads)

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention3D(
            self.dim,
            self.window_size,
            self.num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim, mlp_ratio=mlp_ratio, drop=drop)

        self._cached_mask = None
        self._cached_shape = None

    def forward(self, x):
        """
        x: (B,T,H,W,C)
        """
        B, T, H, W, C = x.shape
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size

        pad_t = (wt - (T % wt)) % wt
        pad_h = (wh - (H % wh)) % wh
        pad_w = (ww - (W % ww)) % ww

        if pad_t or pad_h or pad_w:
            x = F.pad(
                x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t)
            )  # pad W,H,T (channel-last)
        Tp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        shortcut = x
        x = self.norm1(x)

        use_shift = st > 0 or sh > 0 or sw > 0
        if use_shift:
            x = torch.roll(x, shifts=(-st, -sh, -sw), dims=(1, 2, 3))
            key = (Tp, Hp, Wp)
            if (
                (self._cached_shape != key)
                or (self._cached_mask is None)
                or (self._cached_mask.device != x.device)
            ):
                self._cached_mask = compute_shift_mask_3d(
                    Tp, Hp, Wp, self.window_size, self.shift_size, x.device
                )
                self._cached_shape = key
            attn_mask = self._cached_mask
        else:
            attn_mask = None

        x_windows = window_partition_3d(x, self.window_size)  # (B*nW, N, C)

        x_windows = self.attn(x_windows, mask=attn_mask)

        x = window_reverse_3d(x_windows, self.window_size, B, Tp, Hp, Wp, C)

        if use_shift:
            x = torch.roll(x, shifts=(st, sh, sw), dims=(1, 2, 3))

        x = x[:, :T, :H, :W, :].contiguous()

        x = shortcut[:, :T, :H, :W, :] + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalAttnPool(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.score = nn.Linear(E, 1)

    def forward(self, x):  # x: (B, Nsp, T', E)
        w = self.score(x).squeeze(-1)  # (B, Nsp, T')
        w = torch.softmax(w, dim=-1)
        return (x * w.unsqueeze(-1)).sum(dim=2)  # (B, Nsp, E)


class NWPBranchPanguTime3D(nn.Module):
    """
    Input : nwp_y (B, H, W, T, V)   (user: B,h,w,time,channel)
    Treat : time(T) as "height(Z)" in Pangu
    Output: (B, num_sensors, E)  (to match your fusion pipeline)
    """

    def __init__(
        self,
        time_len: int,
        n_vars: int,
        embed_dim: int,
        grid_h: int,
        grid_w: int,
        num_sensors: int,
        patch_size=(2, 4, 4),
        window_size=(2, 6, 12),
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_stats: bool = False,
        stats_mean=None,
        stats_std=None,
        eps: float = 1e-6,
        token_mlp_hidden=None,
        pooling=True,
    ):
        super().__init__()
        self.T = int(time_len)
        self.V = int(n_vars)
        self.E = int(embed_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.num_sensors = int(num_sensors)
        self.patch_size = tuple(patch_size)
        self.window_size = tuple(window_size)
        self.eps = float(eps)

        pt, ph, pw = self.patch_size

        self.use_stats = bool(use_stats)
        if self.use_stats:
            if stats_mean is None or stats_std is None:
                raise ValueError(
                    "stats_mean and stats_std are required when use_stats=True."
                )
            m = torch.as_tensor(stats_mean).detach().clone()
            s = torch.as_tensor(stats_std).detach().clone()

            if m.ndim == 2 and m.shape == (self.T, self.V):
                m = m[None, None, None, :, :]
            if s.ndim == 2 and s.shape == (self.T, self.V):
                s = s[None, None, None, :, :]

            if m.ndim == 1 and m.shape[0] == self.T * self.V:
                m = m[None, None, None, :]
            if s.ndim == 1 and s.shape[0] == self.T * self.V:
                s = s[None, None, None, :]

            self.register_buffer("nwp_mean", m)
            self.register_buffer("nwp_std", s)

        self.patch_embed = nn.Conv3d(
            in_channels=self.V,
            out_channels=self.E,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        pad_t = (pt - (self.T % pt)) % pt
        pad_h = (ph - (self.grid_h % ph)) % ph
        pad_w = (pw - (self.grid_w % pw)) % pw

        self.pad_t, self.pad_h, self.pad_w = int(pad_t), int(pad_h), int(pad_w)
        Tp = _ceil_div(self.T, pt)
        Hp = _ceil_div(self.grid_h, ph)
        Wp = _ceil_div(self.grid_w, pw)
        self.Tp, self.Hp, self.Wp = int(Tp), int(Hp), int(Wp)

        self.N = self.Hp * self.Wp  # tokens

        wt, wh, ww = self.window_size
        shift = (wt // 2, wh // 2, ww // 2)

        dpr = [
            drop_path * i / max(1, (depth - 1)) for i in range(depth)
        ]  # linear drop_path schedule
        blocks = []
        for i in range(depth):
            shift_size = (0, 0, 0) if (i % 2 == 0) else shift
            blocks.append(
                SwinBlock3D(
                    dim=self.E,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_ln = nn.LayerNorm(self.E)

        if token_mlp_hidden is None:
            token_mlp_hidden = max(self.N, self.num_sensors)
        self.token_to_sensor = nn.Sequential(
            nn.Linear(self.N, int(token_mlp_hidden)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(token_mlp_hidden), self.num_sensors),
        )
        self.sensor_ln = nn.LayerNorm(self.E)

        self.pooling = bool(pooling)
        if self.pooling:
            self.temporal_pool = TemporalAttnPool(self.E)
        else:
            self.et2e = nn.Linear(self.E * self.Tp, self.E, bias=True)

    def forward(self, nwp_y: torch.Tensor) -> torch.Tensor:
        """
        nwp_y: (B,H,W,T,V)
        return: (B, num_sensors, E)
        """
        if nwp_y.ndim != 5:
            raise ValueError(f"Expected nwp_y (B,H,W,T,V). Got {tuple(nwp_y.shape)}")
        B, H, W, T, V = nwp_y.shape
        if (H != self.grid_h) or (W != self.grid_w):
            raise ValueError(
                f"Resolution mismatch: got (H,W)=({H},{W}) expected ({self.grid_h},{self.grid_w})"
            )
        if (T != self.T) or (V != self.V):
            raise ValueError(
                f"T/V mismatch: got T={T},V={V} expected T={self.T},V={self.V}"
            )

        x = nwp_y

        if self.use_stats:
            if self.nwp_mean.ndim == 5:  # (1,1,1,T,V)
                x = (x - self.nwp_mean) / (self.nwp_std + self.eps)
            else:  # (1,1,1,T*V)
                x2 = x.reshape(B, H, W, T * V)
                x2 = (x2 - self.nwp_mean) / (self.nwp_std + self.eps)
                x = x2.reshape(B, H, W, T, V)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = x.permute(0, 4, 3, 1, 2).contiguous()

        if self.pad_w or self.pad_h or self.pad_t:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h, 0, self.pad_t))  # pad W,H,T

        x = self.patch_embed(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 3, 1, 4).contiguous()

        if self.pooling:
            x = x.view(B, self.Hp * self.Wp, self.Tp, self.E)
            x = self.temporal_pool(x)
        else:
            x = x.view(B, self.Hp * self.Wp, self.Tp * self.E)
            x = self.et2e(x)

        x = self.out_ln(x)

        x = x.transpose(1, 2).contiguous()
        x = self.token_to_sensor(x)
        x = x.transpose(1, 2).contiguous()
        x = self.sensor_ln(x)
        return x
