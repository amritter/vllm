# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# gfx1100-attn fork, Path A v1.1 experiment: paged flash prefill with
# REGISTER-RESIDENT software prefetch (K/V double buffering in fp8).
#
# Motivation (2026-10-13, after the num_warps 8->32 win, 14.4 -> 21.5
# TF/s): the W32 kernel is linear in ctx (61.3 ms @32k, 183.3 ms @100k,
# exactly proportional) = a steady per-tile latency chain. It is NOT
# DRAM-BW-bound (15.8 GB total traffic = 18 ms floor at 864 GB/s) and
# NOT FLOP-bound (QK+PV ~4.1 TFLOP = 30-60 ms at 70-140 TF/s). With
# num_stages=1 each tile serially does: K load (DRAM/L2 latency) ->
# dequant -> QK dot -> softmax -> V load -> dequant -> PV dot.
# num_stages=2 tried to pipeline via LDS and got WORSE (202-315 ms).
# This variant prefetches K(i+1)/V(i+1) (raw fp8, ~2 VGPR/lane) into
# REGISTERS during tile i's compute: V(i+1) is independent of
# qk(i), K(i+1) overlaps the PV dot.
#
# This file is a standalone A/B experiment; it duplicates
# triton_flash_prefill.py's kernel with the prefetch restructure. If it
# wins the bench it gets promoted; otherwise it stays here as evidence.

import os

import torch

from vllm.triton_utils import tl, triton

PREFILL_BM = int(os.environ.get("VFP_BM", "128"))
PREFILL_BN = int(os.environ.get("VFP_BN", "64"))
PREFILL_WARPS = int(os.environ.get("VFP_WARPS", "32"))
PREFILL_STAGES = int(os.environ.get("VFP_STAGES", "1"))


@triton.jit
def _flash_prefill_fwd_pf(
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
    stride_v_cache_inner,
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
    b_loc_base = B_Loc + cur_batch * stride_b_loc_b

    # ---- phase 1: cached history [0, ctx_len), no causal mask ----
    # Register-resident prefetch: tile i+1's raw fp8 loads are issued
    # while tile i's dequant/dots execute. Tile 0's loads are masked by
    # offs_n < ctx_len, so they are a no-op when ctx_len == 0 and the
    # loop below runs zero times (no enclosing runtime if needed).
    pb0 = tl.load(b_loc_base)
    k_cur = tl.load(
        K_cache
        + pb0 * stride_k_cache_bs
        + cur_kv_head * stride_k_cache_h
        + (offs_d[:, None] // X) * stride_k_cache_d
        + offs_n[None, :] * stride_k_cache_bl
        + (offs_d[:, None] % X),
        mask=offs_n[None, :] < ctx_len,
        other=0.0,
    )
    v_cur = tl.load(
        V_cache
        + pb0 * stride_v_cache_bs
        + cur_kv_head * stride_v_cache_h
        + offs_n[:, None] * stride_v_cache_bl
        + (offs_d[None, :] // X) * stride_v_cache_d
        + (offs_d[None, :] % X) * stride_v_cache_inner,
        mask=offs_n[:, None] < ctx_len,
        other=0.0,
    )
    for start_n in range(0, ctx_len, BN):
        # Issue the NEXT tile's raw loads now (overlap with this tile).
        if start_n + BN < ctx_len:
            nstart = start_n + BN
            pb_n = tl.load(b_loc_base + (nstart // PB))
            in_block_n = (nstart % PB) + offs_n
            if nstart + BN > ctx_len:
                k_nxt = tl.load(
                    K_cache
                    + pb_n * stride_k_cache_bs
                    + cur_kv_head * stride_k_cache_h
                    + (offs_d[:, None] // X) * stride_k_cache_d
                    + in_block_n[None, :] * stride_k_cache_bl
                    + (offs_d[:, None] % X),
                    mask=(nstart + offs_n[None, :]) < ctx_len,
                    other=0.0,
                )
                v_nxt = tl.load(
                    V_cache
                    + pb_n * stride_v_cache_bs
                    + cur_kv_head * stride_k_cache_h
                    + in_block_n[:, None] * stride_v_cache_bl
                    + (offs_d[None, :] // X) * stride_v_cache_d
                    + (offs_d[None, :] % X) * stride_v_cache_inner,
                    mask=(nstart + offs_n[:, None]) < ctx_len,
                    other=0.0,
                )
            else:
                k_nxt = tl.load(
                    K_cache
                    + pb_n * stride_k_cache_bs
                    + cur_kv_head * stride_k_cache_h
                    + (offs_d[:, None] // X) * stride_k_cache_d
                    + in_block_n[None, :] * stride_k_cache_bl
                    + (offs_d[:, None] % X)
                )
                v_nxt = tl.load(
                    V_cache
                    + pb_n * stride_v_cache_bs
                    + cur_kv_head * stride_k_cache_h
                    + in_block_n[:, None] * stride_v_cache_bl
                    + (offs_d[None, :] // X) * stride_v_cache_d
                    + (offs_d[None, :] % X) * stride_v_cache_inner
                )
        else:
            k_nxt = k_cur  # dead values; loop exits after this tile
            v_nxt = v_cur

        if USE_FP8:
            k = (k_cur.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            v = (v_cur.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            k = k_cur
            v = v_cur

        qk = sm_scale * tl.dot(q, k)  # [BM, BN]
        if start_n + BN > ctx_len:
            qk = tl.where(
                (start_n + offs_n[None, :]) < ctx_len, qk, float("-inf")
            )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc)  # v: [BN, D]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        k_cur = k_nxt
        v_cur = v_nxt

    # ---- phase 2: current chunk, dense K/V, causal among rows ----
    # (no prefetch: 25 tiles at 1600/64, short tail)
    for start_n in range(0, query_len, BN):
        valid_kv = (start_n + offs_n[None, :]) < query_len
        off_k = (
            (b_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        if start_n + BN > query_len:
            k = tl.load(K + off_k, mask=valid_kv, other=0.0)
        else:
            k = tl.load(K + off_k)

        qk = sm_scale * tl.dot(q, k)
        attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
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
            v = tl.load(V + off_v, mask=valid_kv.T, other=0.0)
        else:
            v = tl.load(V + off_v)
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


def flash_prefill_paged_fwd_pf(
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
    """Register-prefetch variant of flash_prefill_paged_fwd (A/B only).

    Same interface/semantics as the base kernel.
    """
    batch = b_seq_len.shape[0]
    head = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    num_queries_per_kv = head // num_kv_heads
    D = q.shape[2]
    X = k_cache.shape[4] if k_cache.dim() == 5 else 1
    PB = v_cache.shape[3]
    use_fp8 = k_cache.dtype.is_floating_point and k_cache.element_size() == 1

    grid = (batch, head, triton.cdiv(max_input_len, PREFILL_BM))
    _flash_prefill_fwd_pf[grid](
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
        stride_v_cache_inner=v_cache.stride(4) if v_cache.dim() == 5 else 0,
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
