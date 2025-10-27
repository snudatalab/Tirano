# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import pickle
import math
import copy
from collections import defaultdict
import logging
import os
import sys
import random


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------
def setup_logger(name):
    """
    Create and configure a file logger under ./log/{name}.log.

    Args:
        name (str): Run name for the logger and log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    cur_dir = os.getcwd()
    log_dir = os.path.join(cur_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=os.path.join(log_dir, f'{name}.log'),
        filemode='a'
    )
    logger = logging.getLogger(name)
    return logger


# ---------------------------------------------------------
# Helpers (some projects import these from utils.py)
# ---------------------------------------------------------
def complex_mul(a, b):
    """Element-wise complex multiply when last dim packs (real, imag)."""
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 2
    a_1, a_2 = torch.split(a, dim, dim=-1)
    b_1, b_2 = torch.split(b, dim, dim=-1)
    A = a_1 * b_1 - a_2 * b_2
    B = a_1 * b_2 + a_2 * b_1
    return torch.cat([A, B], dim=-1)


def get_dataset_stat(dataset):
    """
    Expect: dataset/<dataset>/stat.txt with three ints on first line:
      #entities  #relations  #timestamps
    We return:
      (num_e, num_r_with_inverse, num_t)
    where num_r_with_inverse = 2 * #relations (forward + inverse).
    """
    with open(os.path.join('dataset', dataset, 'stat.txt'), 'r') as f:
        for line in f:
            line_ = line.strip().split()
            num_e, num_r, num_t = int(line_[0]), 2 * int(line_[1]), int(line_[2])
            break
    return num_e, num_r, num_t


# ---------------------------------------------------------
# Temporal Neighbor Finder
# ---------------------------------------------------------
class NeighborFinder:
    """
    Temporal neighborhood sampler with multiple strategies.

    Adjacency format:
        - If list: adj[u] = list of (neighbor_entity, relation_id, timestamp)
        - If dict: adj[u] may be missing -> treated as empty list

    Sampling modes (self.sampling):
        0 : uniform random WITH replacement
        1 : first-k         (past-oriented if adjacency sorted ascending by time)
        2 : last-k          (recent-oriented)
        3 : time-distance   ~ exp(-|Δt| / (time_granularity * weight_factor))
        4 : absolute time   ~ (t_i + 1)
        5 : RTNS            ~ exp(-alpha_r * |Δt|^beta), relation-wise alpha
        6 : hybrid          ~ exp(-|Δt|/τ) * alpha_r
        7 : uniform random WITHOUT replacement
       -1 : take ALL neighbors (truncated to last 200000)

    Note:
        All probabilistic modes rely on a safe chooser that handles
        zero/invalid distributions without raising errors.
    """
    def __init__(self,
                 adj,
                 sampling=1,
                 max_time=366 * 24,
                 num_entities=None,
                 weight_factor=1,
                 time_granularity=24,
                 relation2alpha=None,
                 beta=1.0):
        self.time_granularity = int(time_granularity)
        self.sampling = int(sampling)
        self.weight_factor = float(weight_factor)
        self.adj = adj
        self.relation2alpha = relation2alpha if relation2alpha else {}
        self.beta = float(beta)

    # ------------------------------------------------------------------
    # Optional: build offsets (not required for most pipelines)
    # ------------------------------------------------------------------
    def init_off_set(self, adj, max_time, num_entities):
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        off_set_t_l = []

        if isinstance(adj, list):
            assert len(adj) == num_entities
            rng = range(len(adj))
            getter = lambda i: adj[i]
        elif isinstance(adj, dict):
            rng = range(num_entities)
            getter = lambda i: adj.get(i, [])
        else:
            raise TypeError("adj must be list or dict")

        for i in rng:
            curr = getter(i)
            curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            curr_ts = [x[2] for x in curr]
            n_ts_l.extend(curr_ts)

            off_set_l.append(len(n_idx_l))
            off_set_t_l.append(
                [np.searchsorted(curr_ts, cut_time, 'left')
                 for cut_time in range(0, max_time + 1, self.time_granularity)]
            )

        n_idx_l = np.array(n_idx_l, dtype=np.int64)
        n_ts_l = np.array(n_ts_l, dtype=np.int64)
        e_idx_l = np.array(e_idx_l, dtype=np.int64)
        off_set_l = np.array(off_set_l, dtype=np.int64)

        assert len(n_idx_l) == len(n_ts_l)
        assert off_set_l[-1] == len(n_ts_l)

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def set_adj(self, adj):
        self.adj = adj

    # ------------------------------------------------------------------
    # Internal: Safe weighted sampler (no replacement)
    # ------------------------------------------------------------------
    def _safe_weighted_choice(self, total, weights, k):
        """
        Sample k indices in [0, total) without replacement given (possibly zero) weights.
        If k >= total: return a random permutation of all indices (ignore weights).
        If non-zero support < k: take all non-zero-weight indices, fill remainder
        uniformly from the zero-weight set.
        """
        weights = np.asarray(weights, dtype=np.float64)
        if k <= 0 or total <= 0:
            return np.empty((0,), dtype=np.int32)

        # All items requested -> ignore p (p may contain zeros)
        if k >= total:
            return np.random.permutation(total).astype(np.int32)

        w = np.clip(weights, 0.0, None)
        if (not np.isfinite(w).all()) or (w.sum() <= 0.0):
            # invalid or all zeros -> uniform
            return np.random.choice(total, size=k, replace=False).astype(np.int32)

        nz = np.flatnonzero(w > 0.0)
        if len(nz) >= k:
            p = w[nz] / w[nz].sum()
            sub = np.random.choice(len(nz), size=k, replace=False, p=p)
            return nz[sub].astype(np.int32)

        # if positive-weight set smaller than k -> take all of them,
        # then fill remainder from the zero-weight set uniformly
        chosen = nz.astype(np.int32)
        rem = k - len(chosen)
        zero_idx = np.setdiff1d(np.arange(total, dtype=np.int32), chosen, assume_unique=False)
        if rem > 0 and len(zero_idx) > 0:
            extra = np.random.choice(zero_idx, size=rem, replace=False)
            chosen = np.concatenate([chosen, extra]).astype(np.int32)
        return chosen

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def get_temporal_neighbor(self, obj_idx_l, ts_l, num_neighbors=20, rel_q_l=None):
        """
        Sample temporal neighbors per source object at given query times.

        Args:
            obj_idx_l (array-like[int]): Source entity ids, shape (B,).
            ts_l (array-like[int]): Query timestamps, shape (B,).
            num_neighbors (int): Maximum K neighbors per source (padding/truncation applied).
            rel_q_l (array-like[int] or None): Optional query relations (unused in default paths).

        Returns:
            tuple:
                out_ngh_node_batch (np.ndarray[int32]): (B, K) neighbor entity ids, left-padded with -1.
                out_ngh_eidx_batch (np.ndarray[int32]): (B, K) relation ids, left-padded with -1.
                out_ngh_t_batch (np.ndarray[int32]): (B, K) neighbor timestamps, left-padded with 0.
                offset_l (list[[int,int]]): Legacy offsets per row (start,end) in flattened view.
                got_node_emb_l (list[int]): Legacy placeholders (always 0).
        """
        assert len(obj_idx_l) == len(ts_l)
        B = len(obj_idx_l)
        K = int(num_neighbors)

        out_ngh_node_batch = -np.ones((B, K), dtype=np.int32)
        out_ngh_t_batch = np.zeros((B, K), dtype=np.int32)
        out_ngh_eidx_batch = -np.ones((B, K), dtype=np.int32)
        offset_l = []
        got_node_emb_l = []

        if self.sampling == -1:
            full_ngh_node = []
            full_ngh_t = []
            full_ngh_edge = []

        been_through = 0  # for offset bookkeeping

        for i, (obj_idx, cut_time) in enumerate(zip(obj_idx_l, ts_l)):
            got_node_emb_l.append(0)

            # fetch adjacency
            if isinstance(self.adj, dict):
                srt_l = self.adj.get(int(obj_idx), [])
            else:
                srt_l = self.adj[int(obj_idx)]
            if len(srt_l) == 0:
                offset_l.append([been_through, been_through])
                continue

            # unpack arrays
            tmp = np.array(srt_l)  # shape (N, 3)
            ngh_idx = tmp[:, 0].astype(np.int32)
            ngh_eidx = tmp[:, 1].astype(np.int32)
            ngh_ts = tmp[:, 2].astype(np.int32)

            # number we can actually pick
            total = len(ngh_idx)
            if total == 0:
                offset_l.append([been_through, been_through])
                continue

            # ----- sampling modes -----
            if self.sampling == 0:
                # uniform WITH replacement
                sampled_idx = np.random.randint(0, total, size=K).astype(np.int32)
                sampled_idx_sorted = np.sort(sampled_idx)
                out_ngh_node_batch[i, :] = ngh_idx[sampled_idx_sorted]
                out_ngh_t_batch[i, :] = ngh_ts[sampled_idx_sorted]
                out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx_sorted]
                offset_l.append([been_through, been_through + K])
                been_through += K

            elif self.sampling == 1:
                # first-k (past oriented if adj sorted ascending by time)
                k = min(K, total)
                idx = np.arange(0, k, dtype=np.int32)
                fill_start = K - k
                out_ngh_node_batch[i, fill_start:] = ngh_idx[idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[idx]
                offset_l.append([been_through, been_through + k])
                been_through += k

            elif self.sampling == 2:
                # last-k (recent oriented)
                k = min(K, total)
                idx = np.arange(total - k, total, dtype=np.int32)
                fill_start = K - k
                out_ngh_node_batch[i, fill_start:] = ngh_idx[idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[idx]
                offset_l.append([been_through, been_through + k])
                been_through += k

            elif self.sampling == 3:
                # time-distance soft sampling: exp(-|Δt| / (granularity * weight_factor))
                tau = self.time_granularity * max(self.weight_factor, 1e-9)
                delta_t = -np.abs(ngh_ts - int(cut_time)) / float(tau)
                weights = np.exp(delta_t)  # may include very small values
                k = min(K, total)
                sampled_idx = self._safe_weighted_choice(total, weights, k)
                sampled_idx = np.sort(sampled_idx)
                fill_start = K - len(sampled_idx)
                out_ngh_node_batch[i, fill_start:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[sampled_idx]
                offset_l.append([been_through, been_through + len(sampled_idx)])
                been_through += len(sampled_idx)

            elif self.sampling == 4:
                # absolute time bias ~ (t_i + 1)
                weights = (ngh_ts.astype(np.float64) + 1.0)
                k = min(K, total)
                sampled_idx = self._safe_weighted_choice(total, weights, k)
                sampled_idx = np.sort(sampled_idx)
                fill_start = K - len(sampled_idx)
                out_ngh_node_batch[i, fill_start:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[sampled_idx]
                offset_l.append([been_through, been_through + len(sampled_idx)])
                been_through += len(sampled_idx)

            elif self.sampling == 5:
                # RTNS: exp(- alpha_r * |Δt|^beta)
                d = np.abs(ngh_ts - int(cut_time)).astype(np.float64)
                alpha_vals = np.array([self.relation2alpha.get(int(r), 0.01) for r in ngh_eidx], dtype=np.float64)
                weights = np.exp(-alpha_vals * (d ** self.beta))
                k = min(K, total)
                sampled_idx = self._safe_weighted_choice(total, weights, k)
                sampled_idx = np.sort(sampled_idx)
                fill_start = K - len(sampled_idx)
                out_ngh_node_batch[i, fill_start:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[sampled_idx]
                offset_l.append([been_through, been_through + len(sampled_idx)])
                been_through += len(sampled_idx)

            elif self.sampling == 6:
                # NEW: time-distance weight × alpha_r  -> exp(-|Δt|/τ) * alpha_r
                tau = self.time_granularity * max(self.weight_factor, 1e-9)
                w_time = np.exp(-np.abs(ngh_ts - int(cut_time)) / float(tau))  # like sampling==3
                alpha_vals = np.array([self.relation2alpha.get(int(r), 0.01) for r in ngh_eidx], dtype=np.float64)
                weights = w_time * alpha_vals
                k = min(K, total)
                sampled_idx = self._safe_weighted_choice(total, weights, k)
                sampled_idx = np.sort(sampled_idx)
                fill_start = K - len(sampled_idx)
                out_ngh_node_batch[i, fill_start:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[sampled_idx]
                offset_l.append([been_through, been_through + len(sampled_idx)])
                been_through += len(sampled_idx)

            elif self.sampling == 7:
                # NEW: uniform random WITHOUT replacement
                k = min(K, total)
                sampled_idx = np.random.choice(total, size=k, replace=False).astype(np.int32)
                sampled_idx = np.sort(sampled_idx)
                fill_start = K - k
                out_ngh_node_batch[i, fill_start:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, fill_start:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, fill_start:] = ngh_eidx[sampled_idx]
                offset_l.append([been_through, been_through + k])
                been_through += k

            elif self.sampling == -1:
                # use whole neighborhood (truncate to last 200000)
                full_ngh_node.append(ngh_idx[-200000:])
                full_ngh_t.append(ngh_ts[-200000:])
                full_ngh_edge.append(ngh_eidx[-200000:])
                offset_l.append([been_through, been_through])  # legacy
            else:
                raise ValueError(f"invalid input for sampling: {self.sampling}")

        # reshape when using "all neighbors" mode
        if self.sampling == -1:
            if len(full_ngh_edge) == 0:
                return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, offset_l, got_node_emb_l
            max_num_neighbors = max(map(len, full_ngh_edge))
            out_ngh_node_batch = -np.ones((B, max_num_neighbors), dtype=np.int32)
            out_ngh_t_batch = np.zeros((B, max_num_neighbors), dtype=np.int32)
            out_ngh_eidx_batch = -np.ones((B, max_num_neighbors), dtype=np.int32)
            for i in range(len(full_ngh_node)):
                out_ngh_node_batch[i, max_num_neighbors - len(full_ngh_node[i]):] = full_ngh_node[i]
                out_ngh_eidx_batch[i, max_num_neighbors - len(full_ngh_edge[i]):] = full_ngh_edge[i]
                out_ngh_t_batch[i, max_num_neighbors - len(full_ngh_t[i]):] = full_ngh_t[i]

        # return order expected by downstream code:
        # (nodes, relations, times, offset_l, got_node_emb_l)
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, offset_l, got_node_emb_l
