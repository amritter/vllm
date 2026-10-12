# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# gfx1100-attn fork, Path A: fast paged flash prefill for RDNA3 (gfx1100).
#
# Replaces prefix_prefill._fwd_kernel (the "cliff" kernel) for eligible
# calls. Measured root cause of the cliff (docs/vllm-fork.md, 2026-10-11
# profiling): the context history is read from the paged KV cache with
# 32-token gather tiles - one block-table lookup and one scattered
# element offset computation per TOKEN, int64 math, num_stages=1. At
# 100k context the kernel sits at ~0.12% of the DRAM roofline on the
# W7800 (616 s cold 100k prefill, ~25 s cached-recompute tail).
#
# What this kernel does differently:
#   * BN=64-token KV tiles. BN divides the physical block size on this
#     service (PB=1600 = 25 * 64), so every tile lies inside one
#     physical block: ONE scalar block-table load per tile instead of
#     32, and the K/V tile is a single contiguous 2-D load.
#   * Two phases, like the cliff kernel:
#       phase 1: cached history [0, ctx_len) from the paged cache,
#         NO causal mask (every query row sees all of it);
#       phase 2: current chunk from the dense K/V tensors, with the
#         per-row causal mask (row i sees chunk positions <= i).
#   * Flash-2 online softmax per query row (m, l, acc), output written
#     flat like the cliff kernel ((total_q, Hq, D) packing via
#     b_start_loc).
#   * GQA: each query head programs its own kv-head's KV (G redundant
#     KV streams in v1; GQA M-stacking is a later optimization).
#
# v1 eligibility (host gate, see chunked_prefill_paged_decode.py):
# causal, no alibi/sinks/sliding-window/fp8-output, dense K/V provided,
# q in {bf16, fp16}, head dim in {64, 128, 256}, cache x=1 (K 5D with
# last dim 1, V 4D), PB % BN == 0, native cache layout, int block table.

import os

import torch

from vllm.triton_utils import tl, triton

# Tunable via env (tuning sweep, 2026-10-12). Defaults are the measured
# best on the W7800 for 8192 q @ 50k ctx (bench: 1175 ms = 2.0x over the
# cliff kernel's 2399 ms; BM=256 overflows the 64 KB LDS budget):
#   BM=128, BN=64, num_warps=8, num_stages=1.
# BN must divide the physical block size (1600 % 64 == 0) for the
# one-block-table-load-per-tile property.
PREFILL_BM = int(os.environ.get("VFP_BM", "128"))
PREFILL_BN = int(os.environ.get("VFP_BN", "64"))
PREFILL_WARPS = int(os.environ.get("VFP_WARPS", "8"))
PREFILL_STAGES = int(os.environ.get("VFP_STAGES", "1"))

# KV tile size (see file header). Kept separate from the tunable so the
# host-side PB-alignment gate in chunked_prefill_paged_decode stays valid.
_FLASH_PREFILL_BN = PREFILL_BN
# Query rows per program (tunable default above). acc is [BM, D] fp32:
# at BM=128, D=256, num_warps=8 (512 threads) that is 96 VGPR per lane.


@triton.jit
def _flash_prefill_fwd(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_b_loc_b,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    D: tl.constexpr,
    X: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    USE_FP8: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // num_queries_per_kv

    b_start = tl.load(B_Start_Loc + cur_batch)
    b_stop = tl.load(B_Start_Loc + cur_batch + 1)
    query_len = b_stop - b_start
    seq_len = tl.load(B_Seqlen + cur_batch)
    ctx_len = seq_len - query_len

    # Rows with query_len == 1 are handled by the paged 2D decode kernel
    # downstream (same split as the cliff kernel's SKIP_DECODE).
    if SKIP_DECODE and query_len == 1:
        return

    offs_m = start_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BN)
    row_valid = offs_m < query_len

    # q is flat-packed over the batch: (total_q, Hq, D).
    q_ptrs = (
        Q
        + (b_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)  # [BM, D]

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    PB = PHYSICAL_BLOCK_SIZE

    # ---- phase 1: cached history [0, ctx_len), no causal mask ----
    for start_n in range(0, ctx_len, BN):
        tile = start_n // PB
        pb = tl.load(B_Loc + cur_batch * stride_b_loc_b + tile)
        # Contiguous token offsets inside the physical block.
        in_block = (start_n % PB) + offs_n  # [BN]

        last = start_n + BN > ctx_len
        # x-packed K layout: cache is (nb, Hkv, D // x, PB, x) (x == 1
        # for the plain 4D layout, where this degenerates to
        # offs_d * stride_k_cache_d). Consecutive dims inside one x
        # group are unit-stride, and a BN tile of one (d // x) row is
        # BN * x contiguous bytes, so DRAM efficiency is preserved.
        off_k = (
            pb * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // X) * stride_k_cache_d
            + in_block[None, :] * stride_k_cache_bl
            + (offs_d[:, None] % X)
        )
        if last:
            k_load = tl.load(
                K_cache + off_k,
                mask=(start_n + offs_n[None, :]) < ctx_len,
                other=0.0,
            )
        else:
            k_load = tl.load(K_cache + off_k)  # [D, BN]
        if USE_FP8:
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = sm_scale * tl.dot(q, k)  # [BM, BN]
        if last:
            qk = tl.where((start_n + offs_n[None, :]) < ctx_len, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        off_v = (
            pb * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[:, None] * stride_v_cache_d
            + in_block[None, :] * stride_v_cache_bl
        )
        if last:
            v_load = tl.load(
                V_cache + off_v,
                mask=(start_n + offs_n[None, :]) < ctx_len,
                other=0.0,
            )
        else:
            v_load = tl.load(V_cache + off_v)  # [D, BN]
        if USE_FP8:
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        acc = tl.dot(p.to(v.dtype), tl.trans(v), acc=acc)  # v: [D, BN] -> [BN, D]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # ---- phase 2: current chunk, dense K/V, causal among rows ----
    for start_n in range(0, query_len, BN):
        valid_kv = (start_n + offs_n[None, :]) < query_len  # [1, BN]
        off_k = (
            (b_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        if start_n + BN > query_len:
            k = tl.load(K + off_k, mask=valid_kv, other=0.0)  # [D, BN]
        else:
            k = tl.load(K + off_k)  # [D, BN]

        qk = sm_scale * tl.dot(q, k)  # [BM, BN]
        attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        off_v = (
            (b_start + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        if start_n + BN > query_len:
            v = tl.load(V + off_v, mask=valid_kv.T, other=0.0)  # [BN, D]
        else:
            v = tl.load(V + off_v)  # [BN, D]
        acc = tl.dot(p.to(v.dtype), v, acc=acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / (l_i[:, None] + 1e-10)

    out_ptrs = (
        Out
        + (b_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=row_valid[:, None])
    return


def flash_prefill_paged_fwd(
    q,
    k,
    v,
    o,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_input_len,
    k_scale,
    v_scale,
    sm_scale,
    skip_decode: bool = True,
):
    """Fast paged flash prefill (Path A).

    Drop-in replacement for context_attention_fwd on eligible calls:
    same inputs (flat-packed q/k/v/o, paged cache, b_start_loc,
    b_seq_len = full length INCLUDING the current chunk), same output
    semantics. Dense k/v hold the current chunk; the paged cache holds
    the whole sequence (vLLM writes the cache before attention).
    """
    batch = b_seq_len.shape[0]
    head = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    num_queries_per_kv = head // num_kv_heads
    D = q.shape[2]
    # x-packed K cache: (nb, Hkv, D // x, PB, x); x == 1 for 4D caches.
    X = k_cache.shape[4] if k_cache.dim() == 5 else 1
    assert D % X == 0 and X in (1, 2, 4, 8, 16, 32), (
        f"unsupported K-cache packing x={X}, D={D}"
    )
    if k_cache.dim() == 5:
        assert k_cache.shape[2] * X == D and k_cache.shape[3] == v_cache.shape[3]
    PB = v_cache.shape[3]
    use_fp8 = k_cache.dtype.is_floating_point and k_cache.element_size() == 1

    grid = (batch, head, triton.cdiv(max_input_len, PREFILL_BM))
    _flash_prefill_fwd[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        b_loc,
        sm_scale,
        k_scale,
        v_scale,
        b_start_loc,
        b_seq_len,
        o,
        b_loc.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        num_queries_per_kv=num_queries_per_kv,
        D=D,
        X=X,
        BM=PREFILL_BM,
        BN=PREFILL_BN,
        PHYSICAL_BLOCK_SIZE=PB,
        USE_FP8=use_fp8,
        SKIP_DECODE=skip_decode,
        num_warps=PREFILL_WARPS,
        num_stages=PREFILL_STAGES,
    )
    return
