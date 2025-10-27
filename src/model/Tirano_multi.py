import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast

# -------------------------------------------------
# Opt-in global speed-ups 
# -------------------------------------------------
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


# ------------------------------- Utils -------------------------------
def masked_mean_ter(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    y    : (B, C, T, E, R)
    mask : (B, 1, T, E, R) in {0,1}
    returns (B, C) -> masked average over (T,E,R)
    """
    mask = mask.to(y.dtype)
    denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)  # (B,1)
    s = (y * mask).sum(dim=(2, 3, 4)) / denom       # (B,C)
    return s


def masked_mean_time(y: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """
    Compute masked average over time dimension.

    Args:
        y: Tensor of shape (P, C, T).
        mask_t: Binary mask of shape (P, 1, T) with {0,1}.

    Returns:
        Tensor of shape (P, C): masked mean across time.
    """
    mask_t = mask_t.to(y.dtype)
    denom = mask_t.sum(dim=2).clamp_min(1.0)        # (P,1)
    s = (y * mask_t).sum(dim=2) / denom             # (P,C)
    return s


# ------------------------------- Time Encoding -------------------------------
class TimeEncode(nn.Module):
    """Simple harmonic time encoding (cosine banks)."""
    def __init__(self, expand_dim: int):
        super().__init__()
        if expand_dim <= 0:
            self.register_parameter("basis_freq", None)
            self.register_parameter("phase", None)
            self.expand_dim = 0
        else:
            freq = 1.0 / (10.0 ** np.linspace(0, 9, expand_dim))
            self.basis_freq = nn.Parameter(torch.tensor(freq, dtype=torch.float))
            self.phase      = nn.Parameter(torch.zeros(expand_dim))
            self.expand_dim = expand_dim

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ts: 1D tensor of shape (N,) containing time deltas.

        Returns:
            Tensor of shape (N, expand_dim) with cosine features.
            Returns zeros if expand_dim == 0.
        """
        if self.expand_dim == 0:
            return ts.new_zeros((ts.shape[0], 0))
        if ts.dim() == 1:
            ts = ts.unsqueeze(-1)                 # (N,1)
        ts = ts.unsqueeze(-1)                     # (N,1,1)
        map_ts = ts * self.basis_freq.view(1,1,-1) + self.phase.view(1,1,-1)
        return torch.cos(map_ts).squeeze(1)       # (N, d_pe)


# ------------------------------- Dense 3D ST-Temporal Encoder (legacy) -------------------------------
class STTemporalEncoder(nn.Module):
    """
    3D temporal encoder over a dense (T, E, R) grid.

    Block:
        Depthwise time-only Conv3d + pointwise Conv3d, repeated L times.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        k_t: Temporal kernel size (odd only).
        L: Number of stacked depthwise+pointwise blocks.
        dilations: List of temporal dilations per block (len == L).

    Input:
        x: (B, C_in, T, E, R)
        mask (optional): (B, 1, T, E, R) in {0,1}

    Output:
        (B, C_out, T, E, R)
    """
    def __init__(self, c_in: int, c_out: int, k_t: int = 3, L: int = 2, dilations=None):
        super().__init__()
        assert k_t % 2 == 1, "Use odd kernel size for same-padding along time."
        self.c_in = c_in
        self.c_out = c_out
        self.L = L
        if dilations is None:
            dilations = [1] * L
        self.blocks = nn.ModuleList()
        for ell in range(L):
            d = dilations[ell]
            pad_t = (d * (k_t - 1)) // 2
            dw = nn.Conv3d(
                in_channels=c_in, out_channels=c_in,
                kernel_size=(k_t, 1, 1), padding=(pad_t, 0, 0),
                dilation=(d, 1, 1), groups=c_in, bias=False
            )
            pw = nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=1, bias=True)
            self.blocks.append(nn.ModuleDict({
                "dw": dw, "pw": pw, "act": nn.ReLU(inplace=True)
            }))
        self.out_pw = nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=1)

    def forward(self, x, mask=None):  # x: (B, C_in, T, E, R), mask: (B,1,T,E,R)
        """
        Args:
            x: Tensor (B, C_in, T, E, R).
            mask: Optional binary mask (B, 1, T, E, R).

        Returns:
            Tensor (B, C_out, T, E, R) with masked positions zeroed if mask is given.
        """
        orig_dtype = x.dtype
        if torch.is_autocast_enabled():
            with autocast(enabled=False):
                y = x.float()
                for blk in self.blocks:
                    y = blk["dw"](y)
                    y = blk["act"](y)
                    y = blk["pw"](y)
                    y = blk["act"](y)
                y = self.out_pw(y)  # (B, C_out, T, E, R)
                if mask is not None:
                    y = y * mask.to(y.dtype)
        else:
            y = x
            for blk in self.blocks:
                y = blk["dw"](y)
                y = blk["act"](y)
                y = blk["pw"](y)
                y = blk["act"](y)
            y = self.out_pw(y)
            if mask is not None:
                y = y * mask.to(y.dtype)
        return y.to(orig_dtype)


# ------------------------------- Sparse (Pairs×Time) ST-Temporal Encoder (fast path) -------------------------------
class SparseSTTemporalEncoder1D(nn.Module):
    """
    Sparse temporal encoder over (pair, time) sequences.

    Design:
        Depthwise Conv1d (groups=C_in) along time + pointwise Conv1d.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        k_t: Temporal kernel size (odd only).
        L: Number of depthwise+pointwise stages.
        dilations: List of temporal dilations per stage.
        dropout: Dropout rate after final 1×1 Conv1d.

    Input:
        x: (P, C_in, T) where P = number of unique (b, e_slot, r_slot) pairs.

    Output:
        (P, C_out, T)
    """
    def __init__(self, c_in: int, c_out: int, k_t: int = 3, L: int = 2, dilations=None, dropout: float = 0.0):
        super().__init__()
        assert k_t % 2 == 1, "Use odd kernel size for same-padding along time."
        if dilations is None:
            dilations = [1] * L
        layers = []
        for d in dilations:
            pad_t = (d * (k_t - 1)) // 2
            layers += [
                nn.Conv1d(c_in, c_in, kernel_size=k_t, padding=pad_t, dilation=d, groups=c_in, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(c_in, c_in, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.out_pw = nn.Conv1d(c_in, c_out, kernel_size=1, bias=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # x: (P, C_in, T)
        """
        Args:
            x: Tensor (P, C_in, T).

        Returns:
            Tensor (P, C_out, T).
        """
        orig_dtype = x.dtype
        if torch.is_autocast_enabled():
            with autocast(enabled=False):
                x32 = x.float()
                y32 = self.body(x32)          # (P, C_in, T) FP32
                y32 = self.out_pw(y32)        # (P, C_out, T) FP32
        else:
            y32 = self.body(x)
            y32 = self.out_pw(y32)
        y32 = self.drop(y32)
        return y32.to(orig_dtype)


# ------------------------------- Main Model -------------------------------
class Tirano(nn.Module):
    """
    Tirano model.

    Summary:
        - Sparse (pair×time) temporal path with lightweight 1D separable conv (default).
        - Optional dense 3D (T,E,R) grid path for legacy compatibility.
        - Simple neighbor feature aggregation with time-aware features.
        - Flexible decoders: DistMult / ComplEx / Bi-Quaternion.

    Notes:
        Implementation follows the original training/inference behavior exactly.
    """
    def __init__(self,
                 nf,
                 embed_dim: int,
                 num_ent: int,
                 num_rel: int,
                 logger: Optional[object] = None,
                 decoder: str = 'distmult',     # 'distmult' | 'complex' | 'bique' (QuatE)
                 steps: int = 1,
                 device: Union[str, torch.device] = 'cpu',
                 # ST-Temporal Encoder / grid hyper-params
                 cnn_out_dim: int = 64,
                 time_window: int = 960,    # m -> raw window [-m, m]
                 bin_width: int = 120,      # Δ -> T = ceil((2m+1)/Δ)
                 st_L: int = 2,
                 k_t: int = 1,
                 dilations=None,
                 # Slice weighting
                 relation2alpha: dict = None,
                 beta: float = 1.0,
                 slice_temp: float = 2.0,
                 learn_slice_temp: bool = False,
                 time_pe_dim: int = 4,
                 w_clip: float = None,
                 # hr bottleneck
                 hr_c: int = 128,
                 # 3D position encoding (Fourier); 0 disables
                 pos_fourier_dim: int = 4,  # per-axis L => each axis 2L dims
                 # regularization / runtime
                 dropout: float = 0.0,
                 label_smoothing: float = 0.0,
                 use_amp: bool = True,
                 use_sparse_time_encoder: bool = True,  # <<< 빠른 경로 기본값
                 debug_snapshots: bool = False,
                 **kwargs):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.logger = logger
        self.steps = steps
        self.nf = nf

        self.use_amp = bool(use_amp)
        self.use_sparse_time_encoder = bool(use_sparse_time_encoder)

        # ---------------- Loss ----------------
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

        # ---------------- Embeddings (base) ----------------
        # [0..num_ent-1] entities, [num_ent .. num_ent+num_rel-1] relations
        self.symbol_emb = nn.Embedding(num_ent + num_rel, embed_dim)

        # Decoder-specific extra params
        self.decoder = decoder.lower()
        if self.decoder == 'complex':
            self.symbol_emb_im = nn.Embedding(num_ent + num_rel, embed_dim)
            self.to_complex = nn.Linear(embed_dim, 2 * embed_dim)
        elif self.decoder in ('bique', 'quate'):  # quaternion-like
            self.symbol_emb_q_b = nn.Embedding(num_ent + num_rel, embed_dim)
            self.symbol_emb_q_c = nn.Embedding(num_ent + num_rel, embed_dim)
            self.symbol_emb_q_d = nn.Embedding(num_ent + num_rel, embed_dim)
            self.to_quat = nn.Linear(embed_dim, 4 * embed_dim)

        # ---------------- Window / Binning ----------------
        self.m = int(time_window)
        self.bin_width = max(1, int(bin_width))

        # ---------------- Slice-Weighting Params ----------------
        if relation2alpha is not None:
            self.relation2alpha = dict(relation2alpha)
        else:
            self.relation2alpha = getattr(nf, 'relation2alpha', {}) if nf is not None else {}
        self.beta = float(beta if beta is not None else getattr(nf, 'beta', 1.0))

        self.slice_temp = float(slice_temp)
        self.learn_slice_temp = bool(learn_slice_temp)
        if self.learn_slice_temp:
            self.slice_temp_vec = nn.Parameter(torch.full((num_rel,), float(slice_temp)))
        self.register_buffer(
            "alpha_table",
            self._build_alpha_table(num_rel, self.relation2alpha),
            persistent=False
        )

        # ---------------- ST-Temporal Encoders ----------------
        self.hr_c = int(hr_c)
        self.hr_proj = nn.Linear(2 * embed_dim, self.hr_c)
        self.c_in = self.hr_c

        if self.use_sparse_time_encoder:
            self.st1d = SparseSTTemporalEncoder1D(
                c_in=self.c_in, c_out=cnn_out_dim, k_t=k_t, L=st_L, dilations=dilations, dropout=dropout
            )
        else:
            self.st_cnn = STTemporalEncoder(
                c_in=self.c_in, c_out=cnn_out_dim, k_t=k_t, L=st_L, dilations=dilations
            )

        # projection to decoder space
        self.proj_f = nn.Linear(cnn_out_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # ---------------- Aggregation Path ----------------
        self.time_encoder  = TimeEncode(expand_dim=time_pe_dim)
        self.node_emb_proj = nn.Linear(embed_dim + time_pe_dim, embed_dim)

        self.rel_w_past   = nn.Parameter(torch.zeros(num_rel, device=device))
        self.rel_w_future = nn.Parameter(torch.zeros(num_rel, device=device))
        self.w_clip = w_clip

        # ---------------- Position Encoding (Fourier) ----------------
        self.pos_fourier_dim = int(pos_fourier_dim)
        self.use_posenc = self.pos_fourier_dim > 0
        if self.use_posenc:
            self.pos_proj = nn.Linear(6 * self.pos_fourier_dim, self.hr_c)
        self._pos_cache = {}  # {(T,E,R,dtype): (fT, fE, fR)}

        # ---------------- Gated Fusion ----------------
        self.fuse_gate   = nn.Linear(2 * embed_dim, embed_dim)
        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)  # (옵션) 필요 시 사용

        # --------- Buffers / debug ---------
        self.register_buffer("ent_indices", torch.arange(num_ent, dtype=torch.long), persistent=False)
        self.debug_snapshots = bool(debug_snapshots)
        self._snap_dbg_counter = 0

    # ----------------------------------------------------
    def forward(self, batch, num_neighbors=100, max_E=50, max_R=50):
        """
        batch : SimpleCustomBatch with fields (src_idx, rel_idx, target_idx, ts, ...)
        returns: score (B, num_ent)
        """
        B = len(batch.src_idx)
        autocast_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.use_amp and torch.cuda.is_available()))
        )

        with autocast_ctx:
            # 1) Relation-aware neighbor sampling (NF)
            ngh_node, ngh_rel, ngh_time, _, _ = self._get_neighbors(batch, num_neighbors)

            mask_np = (ngh_node != -1)
            b_idx, n_idx = np.where(mask_np)
            if len(b_idx) == 0:
                return self._score_with_zero_context(batch)

            # Flatten valid neighbors
            ent_arr = ngh_node[b_idx, n_idx]    # (TN,)
            rel_arr = ngh_rel [b_idx, n_idx]    # (TN,)
            tim_arr = ngh_time[b_idx, n_idx]    # (TN,)
            t_q     = batch.ts[b_idx]           # (TN,)
            dt_arr  = (tim_arr - t_q).astype(np.int32)

            # 2) Per-neighbor relation-specific past/future weights w_i
            rel_t  = torch.from_numpy(rel_arr).long().to(self.device)    # (TN,)
            dt_t   = torch.from_numpy(dt_arr ).float().to(self.device)   # (TN,)

            dt_t = torch.div(dt_t, 24, rounding_mode='trunc')
            # w_past = 1.0 + 0.5 * torch.tanh(self.rel_w_past  [rel_t])
            # w_fut  = 1.0 + 0.5 * torch.tanh(self.rel_w_future[rel_t])
            w_past, w_fut = 1.0, 1.0
            w_i    = torch.where(dt_t <= 0, w_past, w_fut)               # (TN,)

            # 3) neighbor aggregation (Σ w_i * h_i) / Σ w_i
            ent_ids_t = torch.from_numpy(ent_arr).long().to(self.device)
            ent_emb   = self.symbol_emb(ent_ids_t)                       # (TN, D)
            pe_vec    = self.time_encoder(dt_t)                          # (TN, d_pe)
            node_h    = self.node_emb_proj(torch.cat([ent_emb, pe_vec], dim=1))  # (TN, D)

            dtype = node_h.dtype
            w_i   = w_i.to(dtype)

            b_idx_t = torch.from_numpy(b_idx).long().to(self.device)
            sub_sum = torch.zeros((B, self.embed_dim), device=self.device, dtype=dtype)
            w_sum   = torch.zeros(B, device=self.device, dtype=dtype)
            sub_sum.index_add_(0, b_idx_t, node_h * w_i.unsqueeze(-1))
            w_sum.scatter_add_(0, b_idx_t, w_i)
            w_sum = torch.clamp(w_sum, min=1e-6)
            sub_emb_gnn = torch.tanh(sub_sum / w_sum.unsqueeze(-1))      # (B, D)

            # 4) Temporal encoder: Sparse (Pairs×Time) or Dense grid
            if self.use_sparse_time_encoder:
                h_ctx = self._sparse_temporal_context_fast(
                    batch, B, ent_arr, rel_arr, dt_arr, b_idx,
                    max_E=max_E, max_R=max_R, w_i=w_i
                )  # (B, D)
            else:
                grid, mask = self._build_snapshot_grid_fast(
                    batch, B, ent_arr, rel_arr, dt_arr, b_idx,
                    max_E=max_E, max_R=max_R, w_i=w_i
                )  # grid: (B, C_in, T, E, R), mask: (B,1,T,E,R)
                Y = self.st_cnn(grid, mask=mask)      # (B, C_out, T, E, R)
                z = masked_mean_ter(Y, mask)          # (B, C_out)
                h_ctx = torch.tanh(self.proj_f(z))    # (B, D)
                h_ctx = self.dropout(h_ctx)

            # 5) Gated Fusion: [CNN context || GNN aggregate] -> D
            h_cat = torch.cat([h_ctx, sub_emb_gnn], dim=1)               # (B, 2D)
            g = torch.sigmoid(self.fuse_gate(h_cat))                     # (B, D)
            h_fused = g * h_ctx + (1.0 - g) * sub_emb_gnn

            # 6) Decoder
            score = self._score_from_h(h_fused, batch)
            return score

    # ----------------------------------------------------
    def _get_neighbors(self, batch, num_neighbors):
        """Try relation-aware neighbor sampling if NF supports 'rel_q_l'; else fallback."""
        try:
            return self.nf.get_temporal_neighbor(
                batch.src_idx, batch.ts, rel_q_l=batch.rel_idx, num_neighbors=num_neighbors
            )
        except TypeError:
            return self.nf.get_temporal_neighbor(
                batch.src_idx, batch.ts, num_neighbors=num_neighbors
            )

    # ----------------------------------------------------
    def _build_alpha_table(self, num_rel: int, r2a: dict) -> torch.Tensor:
        """
        Build per-relation alpha lookup.

        Args:
            num_rel: Number of relations.
            r2a: Dict mapping relation id -> alpha.

        Returns:
            Tensor (num_rel,) of alphas with inverse mapping propagation.
        """
        base = num_rel // 2 if (num_rel % 2 == 0) else None
        alpha = [0.01] * num_rel
        for k, v in (r2a or {}).items():
            if 0 <= int(k) < num_rel:
                alpha[int(k)] = float(v)
        if base is not None:
            for r in range(base, num_rel):
                b = r % base
                if alpha[r] == 0.01 and alpha[b] != 0.01:
                    alpha[r] = alpha[b]
        return torch.tensor(alpha, dtype=torch.float)

    # ----------------------------------------------------
    def _compute_slice_weights(self, r_q_vec: torch.Tensor, T: int) -> torch.Tensor:
        """
        Compute soft weights over T temporal slices for each query relation.

        Args:
            r_q_vec: Tensor (B,) relation ids per query.
            T: Number of time bins (slices).

        Returns:
            Tensor (B, T): unnormalized slice weights before within-slice normalization.
        """
        device = r_q_vec.device
        L = -self.m
        # bin centers
        lefts  = torch.arange(T, device=device, dtype=torch.float) * self.bin_width + L
        rights = torch.clamp(lefts + self.bin_width, max=self.m + 1)
        centers = 0.5 * (lefts + (rights - 1.0))       # (T,)
        dist = centers.abs().view(1, T)                 # (1,T)

        alphas = self.alpha_table[r_q_vec].view(-1, 1).to(device)  # (B,1)
        if getattr(self, "learn_slice_temp", False):
            Ts = F.softplus(self.slice_temp_vec[r_q_vec]).view(-1, 1) + 1e-6  # (B,1)
        else:
            Ts = torch.full((r_q_vec.shape[0], 1), float(self.slice_temp), device=device)
        beta = float(self.beta)

        s = torch.exp(-(alphas / Ts) * (dist ** beta))  # (B,T)
        return s

    # ----------------------------------------------------
    def _assign_slots_firstk(self, b_t: torch.Tensor, ids_t: torch.Tensor,
                             budget: int, id_bound: int) -> torch.Tensor:
        """
        Assign compact slots per-batch by first-appearance order.

        Args:
            b_t: Tensor (N,) batch ids per event.
            ids_t: Tensor (N,) raw ids (entity or relation).
            budget: Max number of explicit slots; others map to 'budget'.
            id_bound: Upper bound for id to stabilize hashing.

        Returns:
            Tensor (N,) of slot indices in [0..budget] (budget means "others").
        """
        if budget <= 0:
            return torch.zeros_like(ids_t)

        device = b_t.device
        BIG = int(id_bound + 1)
        key = b_t * BIG + ids_t  # (N,)

        idx_sorted = torch.argsort(key)
        key_sorted = key[idx_sorted]
        first_mask = torch.ones_like(key_sorted, dtype=torch.bool, device=device)
        first_mask[1:] = key_sorted[1:] != key_sorted[:-1]
        uniq_idx_in_sorted = torch.nonzero(first_mask, as_tuple=False).squeeze(-1)
        unique_keys = key_sorted[uniq_idx_in_sorted]               # (U,)
        first_pos   = idx_sorted[uniq_idx_in_sorted]               # original positions
        b_unique    = b_t[first_pos]                               # (U,)

        BIGN = int(key.numel() + 1)
        order2 = torch.argsort(b_unique * BIGN + first_pos)
        b2 = b_unique[order2]

        start_mask = torch.ones_like(b2, dtype=torch.bool, device=device)
        start_mask[1:] = b2[1:] != b2[:-1]  # <<< fix
        group_starts = torch.nonzero(start_mask, as_tuple=False).squeeze(-1)
        group_id = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
        first_index_per_group = group_starts
        rank = torch.arange(order2.numel(), device=device) - first_index_per_group[group_id]
        slot_ordered = torch.where(rank < budget, rank, torch.full_like(rank, budget))

        slots_for_unique = torch.full((unique_keys.numel(),), budget, dtype=torch.long, device=device)
        slots_for_unique[order2] = slot_ordered

        inv = torch.searchsorted(unique_keys, key)  # (N,)
        slots = slots_for_unique[inv]               # (N,)
        return slots

    # ========================= Sparse FAST path =========================
    def _sparse_temporal_context_fast(self, batch, B, ent_arr, rel_arr, dt_arr, b_idx,
                                      max_E=50, max_R=50, w_i: torch.Tensor = None):
        """
        Build sparse (pair,time) sequences and encode them with 1D temporal CNN.

        Args:
            batch: Batch object (used for relation ids of queries).
            B: Batch size.
            ent_arr, rel_arr, dt_arr: Flattened neighbor arrays (numpy).
            b_idx: Batch indices aligned with flattened arrays (numpy).
            max_E, max_R: Slot budgets for entities and relations.
            w_i: Tensor (TN,) per-neighbor weights.

        Returns:
            Tensor (B, D): temporal context features per query.
        """
        assert w_i is not None, "w_i is required."

        device = self.device
        # sizes
        T_prime = 2 * self.m + 1
        T = (T_prime + self.bin_width - 1) // self.bin_width  # ceil
        C_in = self.c_in

        # tensors
        ent_t = torch.from_numpy(ent_arr).long().to(device)
        rel_t = torch.from_numpy(rel_arr).long().to(device)
        dt_l  = torch.from_numpy(dt_arr).long().to(device)
        b_t   = torch.from_numpy(b_idx).long().to(device)

        # τ 
        L_raw = -self.m
        raw_idx = (dt_l - L_raw)                          # shift to [0 .. 2m]
        tau_t   = (raw_idx // self.bin_width)
        valid_mask = (torch.abs(dt_l) <= self.m) & (tau_t >= 0) & (tau_t < T)
        if not bool(valid_mask.any()):
            z_ctx = torch.zeros((B, self.embed_dim), device=device, dtype=self.symbol_emb.weight.dtype)
            return z_ctx

        b_f   = b_t  [valid_mask]   # (N,)
        t_f   = tau_t[valid_mask]   # (N,)
        e_f   = ent_t[valid_mask]   # (N,)
        r_f   = rel_t[valid_mask]   # (N,)
        w_f   = w_i  [valid_mask]   # (N,)

        # (B, τ)별 개수 및 w합 — dtype은 w_f 기준
        flat_bt = b_f * T + t_f
        flat_size = B * T
        one_f = torch.ones_like(w_f, dtype=w_f.dtype, device=device)

        n_counts_flat = torch.zeros(flat_size, dtype=w_f.dtype, device=device)
        n_counts_flat.scatter_add_(0, flat_bt, one_f)
        n_counts = n_counts_flat.view(B, T)  # (B,T)

        w_sums_flat = torch.zeros(flat_size, dtype=w_f.dtype, device=device)
        w_sums_flat.scatter_add_(0, flat_bt, w_f)
        w_sums = w_sums_flat.view(B, T)

        # slice soft weights s[τ] (relation-adaptive + temperature), n[τ]로 재정규화
        rq_tensor = torch.from_numpy(batch.rel_idx).long().to(device)  # (B,)
        s_base = self._compute_slice_weights(rq_tensor, T).to(w_f.dtype)   # (B,T)
        weighted = s_base * n_counts                                       # (B,T)
        denom = weighted.sum(dim=1, keepdim=True).clamp_min(1e-8)
        s = weighted / denom                                               # (B,T)
        zero_mask = (n_counts.sum(dim=1, keepdim=True) == 0.0)
        if bool(zero_mask.any()):
            s = torch.where(zero_mask, torch.full_like(s, 1.0/float(T)), s)

        # hr bottleneck for [E||R]
        ent_vecs = self.symbol_emb(e_f)                                # (N, D)
        rel_vecs = self.symbol_emb(r_f + self.num_ent)                 # (N, D)
        hr_pairs = self.hr_proj(torch.cat([ent_vecs, rel_vecs], dim=1))# (N, hr_c)
        fdtype = hr_pairs.dtype

        # slot  (per-batch first-appearance)
        e_slots = self._assign_slots_firstk(b_f, e_f, max_E - 1, self.num_ent)  # (N,)
        r_slots = self._assign_slots_firstk(b_f, r_f, max_R - 1, self.num_rel)  # (N,)

        if self.use_posenc:
            fT, fE, fR = self._get_pos_banks(T, max_E, max_R, device=device, dtype=fdtype)
            FT = fT[t_f]                                              # (N, 2Lf)
            FE = fE[e_slots.clamp_max(max_E-1)]
            FR = fR[r_slots.clamp_max(max_R-1)]
            F = torch.cat([FT, FE, FR], dim=1)                        # (N, 6Lf)
            hr_pairs = hr_pairs + self.pos_proj(F).to(fdtype)

        # α_i = s[b, τ] * (w_i / sum_w(b, τ))
        eps = torch.tensor(1e-8, device=device, dtype=fdtype)
        s_bt    = s[b_f, t_f].to(fdtype)
        w_denom = (w_sums[b_f, t_f].to(fdtype) + eps)
        alpha   = s_bt * (w_f.to(fdtype) / w_denom)

        # (pair,time) sequence 
        pair_key = b_f * (max_E * max_R) + e_slots * max_R + r_slots      # (N,)
        uniq_keys, pair_idx = torch.unique(pair_key, sorted=True, return_inverse=True)  # (P,), (N,)
        P = uniq_keys.numel()

        pair_b = torch.zeros(P, dtype=b_f.dtype, device=device)
        pair_b.scatter_(0, pair_idx, b_f)

        # sequence tensor: (P, C_in, T) — hr_pairs dtype
        seq    = torch.zeros((P, C_in, T), device=device, dtype=fdtype)
        mask_t = torch.zeros((P, 1,   T), device=device, dtype=fdtype)

        # scatter-add: per (pair, tau)
        flat_pi_tau = (pair_idx * T + t_f).long()               # (N,)
        seq2d = seq.permute(0, 2, 1).contiguous().view(P * T, C_in)   # (P*T, C_in)
        updates = (alpha.view(-1, 1).to(fdtype) * hr_pairs.to(fdtype)) # (N, C_in)
        index2d = flat_pi_tau.view(-1, 1).expand(-1, C_in)
        seq2d.scatter_add_(0, index2d, updates)
        seq = seq2d.view(P, T, C_in).permute(0, 2, 1).contiguous()     # (P, C_in, T)

        # time mask 
        mask_flat = mask_t.view(-1)
        mask_flat.scatter_(0, flat_pi_tau, torch.ones_like(flat_pi_tau, dtype=mask_flat.dtype, device=device))

        # 1D depthwise separable conv
        Y = self.st1d(seq)                                  # (P, C_out, T)
        z_pair = masked_mean_time(Y, mask_t)                # (P, C_out)
        h_ctx_pairs = torch.tanh(self.proj_f(z_pair))       # (P, D)

        # batch-wise average
        sum_ctx = torch.zeros((B, self.embed_dim), device=device, dtype=h_ctx_pairs.dtype)
        cnt_ctx = torch.zeros(B, device=device, dtype=h_ctx_pairs.dtype)
        sum_ctx.index_add_(0, pair_b, h_ctx_pairs)
        cnt_ctx.scatter_add_(0, pair_b, torch.ones_like(pair_b, dtype=h_ctx_pairs.dtype))
        h_ctx = sum_ctx / cnt_ctx.clamp_min(1.0).unsqueeze(-1)    # (B, D)
        h_ctx = self.dropout(h_ctx)
        return h_ctx

    # ----------------------------------------------------
    def _get_pos_banks(self, T: int, E: int, R: int, device, dtype):
        """Cache & return (fT, fE, fR) banks for Fourier features."""
        if not self.use_posenc:
            return None, None, None
        key = (T, E, R, dtype)
        if key in self._pos_cache:
            return self._pos_cache[key]
        Lf = self.pos_fourier_dim

        def fourier_linspace(n):
            if n <= 1:
                vals = torch.zeros(1, device=device, dtype=dtype)
            else:
                vals = torch.linspace(-1.0, 1.0, steps=n, device=device, dtype=dtype)
            k = torch.arange(Lf, device=device, dtype=dtype)
            freq = (2.0 ** k) * math.pi  # (Lf,)
            arg = vals.view(-1, 1) * freq.view(1, -1)  # (n, Lf)
            return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)  # (n, 2Lf)

        fT = fourier_linspace(T)  # (T, 2Lf)
        fE = fourier_linspace(E)  # (E, 2Lf)
        fR = fourier_linspace(R)  # (R, 2Lf)

        self._pos_cache[key] = (fT, fE, fR)
        return fT, fE, fR

    # ========================= Dense grid (legacy) =========================
    def _build_snapshot_grid_fast(self, batch, B, ent_arr, rel_arr, dt_arr, b_idx,
                                  max_E=20, max_R=20, w_i: torch.Tensor = None):
        """
        Construct dense ER-grid and slice weights.

        Args:
            batch: Batch object with query relations.
            B: Batch size.
            ent_arr, rel_arr, dt_arr: Flattened neighbor arrays (numpy).
            b_idx: Batch indices aligned with flattened arrays (numpy).
            max_E, max_R: Slot budgets for entities and relations.
            w_i: Tensor (TN,) per-neighbor weights.

        Returns:
            grid: Tensor (B, C_in, T, E, R) with slice-normalized updates.
            mask: Tensor (B, 1, T, E, R) indicating filled cells.
        """
        assert w_i is not None, "w_i is required for within-slice normalization"

        device = self.device
        T_prime = 2 * self.m + 1
        T = (T_prime + self.bin_width - 1) // self.bin_width
        C_in = self.c_in

        grid = torch.zeros((B, C_in, T, max_E, max_R), device=device)
        mask = torch.zeros((B, 1,   T, max_E, max_R), device=device)

        L_raw = -self.m
        ent_t = torch.from_numpy(ent_arr).long().to(device)
        rel_t = torch.from_numpy(rel_arr).long().to(device)
        dt_t  = torch.from_numpy(dt_arr).long().to(device)
        b_t   = torch.from_numpy(b_idx).long().to(device)

        raw_idx = (dt_t - L_raw)
        tau_t   = (raw_idx // self.bin_width).clamp(min=-1, max=T)
        valid_mask = (torch.abs(dt_t) <= self.m) & (tau_t >= 0) & (tau_t < T)
        if not bool(valid_mask.any()):
            return grid, mask

        b_f   = b_t  [valid_mask]
        t_f   = tau_t[valid_mask]
        e_f   = ent_t[valid_mask]
        r_f   = rel_t[valid_mask]
        w_f   = w_i  [valid_mask]

        flat_bt = b_f * T + t_f
        flat_size = B * T
        one_f = torch.ones_like(w_f, dtype=w_f.dtype, device=device)

        n_counts_flat = torch.zeros(flat_size, dtype=w_f.dtype, device=device)
        n_counts_flat.scatter_add_(0, flat_bt, one_f)
        n_counts = n_counts_flat.view(B, T)

        w_sums_flat = torch.zeros(flat_size, dtype=w_f.dtype, device=device)
        w_sums_flat.scatter_add_(0, flat_bt, w_f)
        w_sums = w_sums_flat.view(B, T)

        rq_tensor = torch.from_numpy(batch.rel_idx).long().to(device)  # (B,)
        s_base = self._compute_slice_weights(rq_tensor, T).to(w_f.dtype)  # (B,T)
        weighted = s_base * n_counts
        denom = weighted.sum(dim=1, keepdim=True).clamp_min(1e-8)
        s = weighted / denom
        zero_mask = (n_counts.sum(dim=1, keepdim=True) == 0.0)
        if bool(zero_mask.any()):
            s = torch.where(zero_mask, torch.full_like(s, 1.0/float(T)), s)

        ent_vecs = self.symbol_emb(e_f)                                # (N, D)
        rel_vecs = self.symbol_emb(r_f + self.num_ent)                 # (N, D)
        hr_pairs = self.hr_proj(torch.cat([ent_vecs, rel_vecs], dim=1))# (N, hr_c)
        fdtype = hr_pairs.dtype

        # grid/mask를 hr_pairs dtype으로 재생성
        grid = torch.zeros((B, C_in, T, max_E, max_R), device=device, dtype=fdtype)
        mask = torch.zeros((B, 1,   T, max_E, max_R), device=device, dtype=fdtype)

        e_slots = self._assign_slots_firstk(b_f, e_f, max_E - 1, self.num_ent)
        r_slots = self._assign_slots_firstk(b_f, r_f, max_R - 1, self.num_rel)

        eps = torch.tensor(1e-8, device=device, dtype=fdtype)
        s_bt   = s[b_f, t_f].to(fdtype)
        w_denom= (w_sums[b_f, t_f].to(fdtype) + eps)
        alpha  = s_bt * (w_f.to(fdtype) / w_denom)

        # scatter-add into grid
        flat_index = (((b_f * T) + t_f) * max_E + e_slots).long()
        flat_index = (flat_index * max_R + r_slots).long()
        grid_flat = grid.view(-1, C_in)
        updates = alpha.view(-1, 1).to(fdtype) * hr_pairs.to(fdtype)
        index2d = flat_index.view(-1, 1).expand(-1, C_in)
        grid_flat.scatter_add_(0, index2d, updates)

        # mask
        mask_flat = mask.view(-1)
        mask_flat.scatter_(0, flat_index, torch.ones_like(flat_index, dtype=mask_flat.dtype, device=device))

        if (getattr(self, "debug_snapshots", False) or os.getenv("TIRANO_DEBUG_SNAPSHOTS", "")):
            self._snap_dbg_counter += 1
            if self._snap_dbg_counter <= 3:
                try:
                    tau_has_any = (n_counts > 0)                     # (B, T)
                    filled_tau_per_b = tau_has_any.sum(dim=1)        # (B,)
                    print(f"[GRID] grid={tuple(grid.shape)}, T={T}, m={self.m}, Δ={self.bin_width}")
                    max_show = min(4, B)
                    for bb in range(max_show):
                        active_idx = torch.nonzero(tau_has_any[bb], as_tuple=False).squeeze(-1).tolist()
                        print(f"[GRID] b={bb}: active_slices={int(filled_tau_per_b[bb])}, tau_idx={active_idx}")
                except Exception as e:
                    print(f"[GRID][WARN] debug print failed: {e}")
        return grid, mask

    # ---------------------- Decoders ----------------------
    def _score_from_h(self, h_sq: torch.Tensor, batch):
        dec = self.decoder
        if dec == 'distmult':
            return self.distmult_score_from_h(h_sq, batch)
        elif dec == 'complex':
            return self.complex_score_from_h(h_sq, batch)
        elif dec in ('bique', 'quate'):
            return self.bique_score_from_h(h_sq, batch)
        else:
            raise ValueError(f"Unknown decoder: {self.decoder}")

    def distmult_score_from_h(self, h_sq: torch.Tensor, batch):
        r_ids = torch.from_numpy(batch.rel_idx).long().to(self.device) + self.num_ent
        dtype = h_sq.dtype
        r_emb = self.symbol_emb(r_ids).to(dtype)                # (B,D)
        match = h_sq * r_emb                                    # (B,D)
        ent_weight = self.symbol_emb.weight[:self.num_ent].to(dtype)  # (num_ent, D)
        return match @ ent_weight.T                             # (B, num_ent)

    def complex_score_from_h(self, h_sq: torch.Tensor, batch):
        B, D = h_sq.shape
        z = self.to_complex(h_sq)          # (B, 2D) — dtype 
        h_r, h_i = z.chunk(2, dim=1)       # (B, D)
        dtype = h_sq.dtype

        r_ids = torch.from_numpy(batch.rel_idx).long().to(self.device) + self.num_ent
        r_r = self.symbol_emb(r_ids).to(dtype)                  # (B, D)
        r_i = self.symbol_emb_im(r_ids).to(dtype)               # (B, D)

        a = h_r * r_r - h_i * r_i                               # (B, D)
        b = h_r * r_i + h_i * r_r                               # (B, D)

        E_r = self.symbol_emb.weight[:self.num_ent].to(dtype)   # (N, D)
        E_i = self.symbol_emb_im.weight[:self.num_ent].to(dtype)# (N, D)
        score = a @ E_r.T + b @ E_i.T                           # (B, N)
        return score

    def bique_score_from_h(self, h_sq: torch.Tensor, batch):
        B, D = h_sq.shape
        z = self.to_quat(h_sq)                      # (B, 4D)
        qa, qb, qc, qd = z.chunk(4, dim=1)         # (B, D) x4
        dtype = h_sq.dtype

        r_ids = torch.from_numpy(batch.rel_idx).long().to(self.device) + self.num_ent
        ra = self.symbol_emb     (r_ids).to(dtype)  # (B, D)
        rb = self.symbol_emb_q_b (r_ids).to(dtype)
        rc = self.symbol_emb_q_c (r_ids).to(dtype)
        rd = self.symbol_emb_q_d (r_ids).to(dtype)

        eps = torch.tensor(1e-6, device=self.device, dtype=dtype)
        rnorm = torch.sqrt(ra*ra + rb*rb + rc*rc + rd*rd + eps)
        ra, rb, rc, rd = ra/rnorm, rb/rnorm, rc/rnorm, rd/rnorm

        ha, hb, hc, hd = qa, qb, qc, qd
        pa = ha*ra - hb*rb - hc*rc - hd*rd
        pb = ha*rb + hb*ra + hc*rd - hd*rc
        pc = ha*rc - hb*rd + hc*ra + hd*rb
        pd = ha*rd + hb*rc - hc*rb + hd*ra

        Ea = self.symbol_emb     .weight[:self.num_ent].to(dtype)
        Eb = self.symbol_emb_q_b .weight[:self.num_ent].to(dtype)
        Ec = self.symbol_emb_q_c .weight[:self.num_ent].to(dtype)
        Ed = self.symbol_emb_q_d .weight[:self.num_ent].to(dtype)
        score = pa @ Ea.T + pb @ Eb.T + pc @ Ec.T + pd @ Ed.T
        return score

    # ----------------------------------------------------
    def _score_with_zero_context(self, batch):
        B = len(batch.src_idx)
        return torch.zeros((B, self.num_ent), device=self.device)

    # ----------------------------------------------------
    def loss(self, score, obj_t):
        return self.loss_func(score, obj_t)

    # ====================================================
    # ============ Pretrained Init Utilities =============
    # ====================================================
    @torch.no_grad()
    def init_from_pretrained(
        self,
        ent_emb: Union[np.ndarray, torch.Tensor, Tuple],
        rel_emb: Union[np.ndarray, torch.Tensor, Tuple],
        kind: str,  # 'distmult'|'rgcn'|'compgcn'|'complex'|'rotate'|'quaternion'|'quate'|'bique'
        ent_id_map: Optional[np.ndarray] = None,
        rel_id_map: Optional[np.ndarray] = None,
        generate_inverse: Optional[bool] = None,
        set_identity_heads: bool = True,
        device: Optional[torch.device] = None,
    ):
        device = device or self.symbol_emb.weight.device
        kind = (kind or '').lower()

        def ensure_t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        def project_to_dim(x: torch.Tensor, out_dim: int, W: Optional[torch.Tensor]):
            if x.size(1) == out_dim:
                return x, W
            if W is None:
                W = torch.empty(x.size(1), out_dim, device=device)
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            return x @ W, W

        def maybe_remap(x: torch.Tensor, idx: Optional[np.ndarray]) -> torch.Tensor:
            if idx is None:
                return x
            ids = torch.as_tensor(idx, dtype=torch.long, device=device)
            return x.index_select(0, ids)

        def conj_complex(re: torch.Tensor, im: torch.Tensor):
            return re, -im

        def conj_quat(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
            return a, -b, -c, -d

        complex_like = kind in ('complex', 'rotate')
        quat_like    = kind in ('quaternion', 'quate', 'bique')

        if complex_like:
            ent_re, ent_im = [ensure_t(a) for a in ent_emb]
            rel_re, rel_im = [ensure_t(a) for a in rel_emb]
            E_pre, D_pre = ent_re.size(0), ent_re.size(1)
            R_pre = rel_re.size(0)
        elif quat_like:
            ent_a, ent_b, ent_c, ent_d = [ensure_t(a) for a in ent_emb]
            rel_a, rel_b, rel_c, rel_d = [ensure_t(a) for a in rel_emb]
            E_pre, D_pre = ent_a.size(0), ent_a.size(1)
            R_pre = rel_a.size(0)
        else:
            ent = ensure_t(ent_emb)
            rel = ensure_t(rel_emb)
            E_pre, D_pre = ent.size(0), ent.size(1)
            R_pre = rel.size(0)

        assert self.num_ent <= E_pre if ent_id_map is None else True, \
            "..."

        if generate_inverse is None:
            generate_inverse = (self.num_rel == 2 * R_pre)

        if complex_like:
            rel_re_full, rel_im_full = rel_re, rel_im
            if generate_inverse:
                ri_re, ri_im = conj_complex(rel_re, rel_im)
                rel_re_full = torch.cat([rel_re, ri_re], dim=0)
                rel_im_full = torch.cat([rel_im, ri_im], dim=0)
            rel_re_full = maybe_remap(rel_re_full, rel_id_map)
            rel_im_full = maybe_remap(rel_im_full, rel_id_map)
        elif quat_like:
            rel_a_full, rel_b_full, rel_c_full, rel_d_full = rel_a, rel_b, rel_c, rel_d
            if generate_inverse:
                ra, rb, rc, rd = conj_quat(rel_a, rel_b, rel_c, rel_d)
                rel_a_full = torch.cat([rel_a, ra], dim=0)
                rel_b_full = torch.cat([rel_b, rb], dim=0)
                rel_c_full = torch.cat([rel_c, rc], dim=0)
                rel_d_full = torch.cat([rel_d, rd], dim=0)
            rel_a_full = maybe_remap(rel_a_full, rel_id_map)
            rel_b_full = maybe_remap(rel_b_full, rel_id_map)
            rel_c_full = maybe_remap(rel_c_full, rel_id_map)
            rel_d_full = maybe_remap(rel_d_full, rel_id_map)
        else:
            rel_full = rel
            if generate_inverse:
                rel_full = torch.cat([rel, rel], dim=0)
            rel_full = maybe_remap(rel_full, rel_id_map)

        if complex_like:
            ent_re_cur = maybe_remap(ent_re, ent_id_map)
            ent_im_cur = maybe_remap(ent_im, ent_id_map)
        elif quat_like:
            ent_a_cur = maybe_remap(ent_a, ent_id_map)
            ent_b_cur = maybe_remap(ent_b, ent_id_map)
            ent_c_cur = maybe_remap(ent_c, ent_id_map)
            ent_d_cur = maybe_remap(ent_d, ent_id_map)
        else:
            ent_cur = maybe_remap(ent, ent_id_map)

        W_shared = None
        D = self.embed_dim

        if complex_like:
            ent_re_cur, W_shared = project_to_dim(ent_re_cur, D, W_shared)
            ent_im_cur, W_shared = project_to_dim(ent_im_cur, D, W_shared)
            rel_re_full, W_shared = project_to_dim(rel_re_full, D, W_shared)
            rel_im_full, W_shared = project_to_dim(rel_im_full, D, W_shared)
        elif quat_like:
            ent_a_cur, W_shared = project_to_dim(ent_a_cur, D, W_shared)
            ent_b_cur, W_shared = project_to_dim(ent_b_cur, D, W_shared)
            ent_c_cur, W_shared = project_to_dim(ent_c_cur, D, W_shared)
            ent_d_cur, W_shared = project_to_dim(ent_d_cur, D, W_shared)
            rel_a_full, W_shared = project_to_dim(rel_a_full, D, W_shared)
            rel_b_full, W_shared = project_to_dim(rel_b_full, D, W_shared)
            rel_c_full, W_shared = project_to_dim(rel_c_full, D, W_shared)
            rel_d_full, W_shared = project_to_dim(rel_d_full, D, W_shared)
        else:
            ent_cur, W_shared = project_to_dim(ent_cur, D, W_shared)
            rel_full, W_shared = project_to_dim(rel_full, D, W_shared)

        # copy into tables
        if complex_like:
            self.symbol_emb.weight[:self.num_ent].copy_(ent_re_cur[:self.num_ent])
            self.symbol_emb.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_re_full[:self.num_rel])
            if hasattr(self, "symbol_emb_im"):
                self.symbol_emb_im.weight[:self.num_ent].copy_(ent_im_cur[:self.num_ent])
                self.symbol_emb_im.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_im_full[:self.num_rel])
        elif quat_like:
            self.symbol_emb.weight[:self.num_ent].copy_(ent_a_cur[:self.num_ent])
            self.symbol_emb.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_a_full[:self.num_rel])
            if hasattr(self, "symbol_emb_q_b"):
                self.symbol_emb_q_b.weight[:self.num_ent].copy_(ent_b_cur[:self.num_ent])
                self.symbol_emb_q_b.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_b_full[:self.num_rel])
            if hasattr(self, "symbol_emb_q_c"):
                self.symbol_emb_q_c.weight[:self.num_ent].copy_(ent_c_cur[:self.num_ent])
                self.symbol_emb_q_c.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_c_full[:self.num_rel])
            if hasattr(self, "symbol_emb_q_d"):
                self.symbol_emb_q_d.weight[:self.num_ent].copy_(ent_d_cur[:self.num_ent])
                self.symbol_emb_q_d.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_d_full[:self.num_rel])
        else:
            self.symbol_emb.weight[:self.num_ent].copy_(ent_cur[:self.num_ent])
            self.symbol_emb.weight[self.num_ent:self.num_ent+self.num_rel].copy_(rel_full[:self.num_rel])
            if hasattr(self, "symbol_emb_im"):
                self.symbol_emb_im.weight.zero_()
            if hasattr(self, "symbol_emb_q_b"):
                self.symbol_emb_q_b.weight.zero_()
            if hasattr(self, "symbol_emb_q_c"):
                self.symbol_emb_q_c.weight.zero_()
            if hasattr(self, "symbol_emb_q_d"):
                self.symbol_emb_q_d.weight.zero_()

        if set_identity_heads:
            if hasattr(self, "to_complex"):
                self.to_complex.weight.zero_()
                self.to_complex.bias.zero_()
                Dm = self.embed_dim
                self.to_complex.weight[:Dm, :Dm].copy_(torch.eye(Dm, device=device))
            if hasattr(self, "to_quat"):
                self.to_quat.weight.zero_()
                self.to_quat.bias.zero_()
                Dm = self.embed_dim
                self.to_quat.weight[:Dm, :Dm].copy_(torch.eye(Dm, device=device))

    # ----------------------------------------------------
    @staticmethod
    def build_optimizer_param_groups(
        model: "Tirano",
        lr_emb: float = 1e-4, wd_emb: float = 0.0,
        lr_other: float = 5e-4, wd_other: float = 1e-4
    ):
        emb_params = [model.symbol_emb.weight]
        if hasattr(model, "symbol_emb_im"):
            emb_params.append(model.symbol_emb_im.weight)
        if hasattr(model, "symbol_emb_q_b"):
            emb_params.extend([
                model.symbol_emb_q_b.weight,
                model.symbol_emb_q_c.weight,
                model.symbol_emb_q_d.weight
            ])
        emb_params_ids = set(id(p) for p in emb_params)
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in emb_params_ids]
        return [
            {'params': emb_params, 'lr': lr_emb, 'weight_decay': wd_emb},
            {'params': other_params, 'lr': lr_other, 'weight_decay': wd_other},
        ]

    # ----------------------------------------------------
    def bias_fuse_gate_towards_cnn(self, bias_value: float = 1.5):
        with torch.no_grad():
            nn.init.constant_(self.fuse_gate.bias, bias_value)
