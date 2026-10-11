# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd
from .triton_decode_attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_grouped_multiq,
)

logger = init_logger(__name__)


# K-split flash-decode tuning for gfx1100 (W7800: 52 CUs). Measured on
# 2026-10-11 (docs/vllm-fork.md, containers repo): 100k-context fp8 KV
# read in 1.14 ms with 52 splits (4 kv heads x 13) vs 106.7 ms for the
# single-pass context kernel. Split length 8192 keeps each split's KV
# tile small enough for the LDS budget at head_dim 256.
_FLASH_DECODE_MAX_SPLITS = 52
_FLASH_DECODE_SPLIT_LEN = 8192
_FLASH_DECODE_MAX_LQ = 8

float8_info = torch.finfo(current_platform.fp8_dtype())


def has_native_kv_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Return whether KV cache blocks can use the native ROCm pairing.

    The native reshape_and_cache writer assumes packed blocks. If cache update
    needs reshape_and_cache_flash for a stride-padded hybrid layout, decode
    should use the matching Triton path too.
    """
    return (
        key_cache.stride(0) == key_cache.shape[1:].numel()
        and value_cache.stride(0) == value_cache.shape[1:].numel()
    )


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale_inv,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.where(float("-inf") < M, 1.0, 0.0)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # iterate through tiles
    for j in range(0, num_blocks):
        start_n = j * BLOCK_SIZE
        # Calculate the logical location within a non-standard physical block,
        # such as 544 in Qwen/Qwen3-Next-80B-A3B-Thinking.
        # Supports non-contiguous mapping
        # from logical blocks to physical blocks
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        # Vectorized loading of physical block IDs
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        # 5D addressing logic of K
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # 4D addressing logic of V (Slot is innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        # Only the final tile can straddle seq_len. Slots >= seq_len are
        # unwritten KV cache that may hold NaN/garbage; they are score-masked
        # below, but 0 * NaN = NaN would still poison the output, so mask them
        # out of the K/V loads too. Earlier tiles are fully written, so they
        # use the cheaper token-uniform dim_mask (matching the pre-0.25.0 fast
        # path) and skip the per-token predicate entirely.
        # K : (HEAD_SIZE, BLOCK_SIZE), V : (BLOCK_SIZE, HEAD_SIZE)
        if j == num_blocks - 1:
            kv_load_mask = abs_token_idx < seq_len
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & kv_load_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
        else:
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(p, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(p.to(V.dtype), V)

    # epilogue
    acc = acc / (L[:, None] + 1e-10)
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


def _try_flash_decode(
    query,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    sm_scale,
    alibi_slopes,
    sliding_window,
    sinks,
    is_block_table_ptr,
    causal,
    output_scale=None,
) -> bool:
    """K-split flash-decode attention for short-query steps (decode + MTP).

    Replaces both the context prefill kernel and the paged decode kernel:
    the new tokens are already in the paged KV cache, so attention over
    the full cached sequence covers them. Each query token becomes its
    own sub-query (batch' = batch * Lq) with an extended seq_len, so the
    standard q_len=1 K-split flash-decode kernel covers up to Lq query
    tokens per request (MTP verification) without kernel changes.

    Why: the single-pass context/paged kernels assign the full context
    to 4-24 thread blocks (batch x kv heads), leaving ~50 of 52 CUs
    idle; at 100k context that costs ~100 ms/layer (0.12% of the KV
    bandwidth roofline). Flash decode splits the context across CUs and
    reads it at roofline speed (1.14 ms/layer at 100k).

    Returns True if the fast path was taken (output is written), False if
    the caller must run its standard implementation.
    """
    if output_scale is not None:
        # no fused output quantization in the flash-decode path
        return False
    if not causal:
        return False
    if alibi_slopes is not None or sinks is not None:
        return False
    if sliding_window and sliding_window > 0:
        return False
    if max_query_len < 1 or max_query_len > _FLASH_DECODE_MAX_LQ:
        return False
    if query.dtype not in (torch.half, torch.bfloat16):
        return False
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    if head_size > 256:
        return False
    # KV cache: the real ROCm paged layout (strided, token innermost):
    #   K: (num_blocks, num_kv_heads, head_dim / x, PAGE_SIZE, x)
    #   V: (num_blocks, num_kv_heads, head_dim, PAGE_SIZE)
    # PAGE_SIZE is the physical block size: 16 for standard models, but
    # inflated for hybrid linear-attention models (this one: 1600). The
    # block table has one row per physical block; the kernel is layout
    # agnostic via the strides the launcher passes.
    if key_cache.ndim != 5 or value_cache.ndim != 4:
        return False
    real_block_size = value_cache.shape[3]
    if real_block_size < 2 or key_cache.shape[3] != real_block_size:
        return False
    if key_cache.shape[1] != value_cache.shape[1]:
        return False
    if key_cache.shape[2] * key_cache.shape[4] != head_size:
        return False
    if value_cache.shape[2] != head_size:
        return False
    num_seqs = len(seq_lens)
    num_tokens = query.shape[0]
    # Uniform query length across requests (shape math only, no sync).
    if num_seqs == 0 or num_tokens != num_seqs * max_query_len:
        return False

    lq = max_query_len
    batch = num_seqs
    n_sub = batch * lq

    # fp8 KV view (the kernel reads the element dtype)
    k_buf = key_cache
    v_buf = value_cache
    if "fp8" in kv_cache_dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            t = current_platform.fp8_dtype()
        else:
            t = torch.float8_e5m2
        k_buf = k_buf.view(t)
        v_buf = v_buf.view(t)

    # Query rows are already request-major: row b' = r*lq + i. For
    # lq == 1 the sub-query batch IS the request batch (identity);
    # for lq >= 2 the multiq stage1 consumes the UNEXPANDED q
    # (n_sub = batch*lq rows) plus the request-level block table and
    # seq lens and streams each request's KV once for all its lq
    # sub-queries (~lq less KV traffic at long context). Query token i
    # of request r sits at KV offset seq_lens[r] - lq + i and causally
    # attends to everything up to and including itself (seq_lens
    # includes the new tokens, which are already in the paged cache).
    if is_block_table_ptr:
        kv_element_size = key_cache.element_size()
        block_byte_stride = key_cache.stride(0) * kv_element_size
        bt = ((block_table - key_cache.data_ptr()) // block_byte_stride).to(
            torch.int32
        )
    else:
        bt = block_table.to(torch.int32)
    seq_i32 = seq_lens.to(torch.int32)

    # K-split count: 4 kv heads x splits ~ fill the 52 CUs
    splits = min(
        _FLASH_DECODE_MAX_SPLITS, max(1, (max_seq_len + _FLASH_DECODE_SPLIT_LEN - 1)
                                       // _FLASH_DECODE_SPLIT_LEN)
    )

    o = output.view(n_sub, num_query_heads, head_size)
    lse = torch.empty(n_sub, num_query_heads, device=query.device,
                      dtype=torch.float32)
    # mid is indexed (sub-query row, head, split); the stage1 kernels
    # skip the store for empty splits, so zero the buffer and mark the
    # lse column -inf: the stage2 logsumexp merge then weights empty
    # splits exactly 0 (an uninit read would corrupt the merge for
    # short requests batched with long ones).
    mid = torch.empty(n_sub, num_query_heads, splits, head_size + 1,
                      device=query.device, dtype=torch.float32)
    mid.zero_()
    mid[..., head_size] = float("-inf")
    k_arg = (torch.tensor(1.0, device=query.device)
             if k_scale is None else k_scale)
    v_arg = (torch.tensor(1.0, device=query.device)
             if v_scale is None else v_scale)
    if lq == 1:
        decode_attention_fwd_grouped(
            q=query,
            k_buffer=k_buf,
            v_buffer=v_buf,
            o=o,
            lse=lse,
            req_to_token=bt,
            b_seq_len=seq_i32,
            attn_logits=mid,
            num_kv_splits=splits,
            sm_scale=sm_scale,
            page_size=real_block_size,
            logit_cap=0.0,
            k_scale=k_arg,
            v_scale=v_arg,
        )
    else:
        decode_attention_fwd_grouped_multiq(
            q=query,
            k_buffer=k_buf,
            v_buffer=v_buf,
            o=o,
            lse=lse,
            req_to_token=bt,
            b_seq_len=seq_i32,
            attn_logits=mid,
            num_kv_splits=splits,
            sm_scale=sm_scale,
            page_size=real_block_size,
            lq=lq,
            k_scale=k_arg,
            v_scale=v_arg,
        )
    return True


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    # Optional tensor for sinks
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
):
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if _try_flash_decode(
        query=query,
        output=output,
        kv_cache_dtype=kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        max_query_len=max_query_len,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window,
        sinks=sinks,
        is_block_table_ptr=is_block_table_ptr,
        causal=causal,
        output_scale=output_scale,
    ):
        return output

    if _try_mixed_decode_prefill_split(
        query=query,
        key=key,
        value=value,
        output=output,
        kv_cache_dtype=kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        max_query_len=max_query_len,
        k_scale=k_scale,
        v_scale=v_scale,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        sinks=sinks,
        is_block_table_ptr=is_block_table_ptr,
        causal=causal,
        output_scale=output_scale,
    ):
        return output

    _standard_context_decode_path(
        query=query,
        key=key,
        value=value,
        output=output,
        kv_cache_dtype=kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        max_query_len=max_query_len,
        k_scale=k_scale,
        v_scale=v_scale,
        alibi_slopes=alibi_slopes,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        sinks=sinks,
        is_block_table_ptr=is_block_table_ptr,
        causal=causal,
        output_scale=output_scale,
    )
    return output




def _standard_context_decode_path(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
):
    """The standard (pre-A') implementation, extracted so a request
    subset can run it in isolation (see _try_mixed_decode_prefill_split).

    Row partitioning is exclusive: the context kernel handles the cached
    prefix of requests with query_len > 1 (skip_decode=True), the paged
    2D kernel handles the full KV of requests with query_len == 1
    (filter_by_query_len=True). No row is written twice, so a subset's
    q/k/v/output slices can be computed independently.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0
    use_alibi_slopes = alibi_slopes is not None

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
            fp8_out_scale=output_scale,
            sinks=sinks,
            causal=causal,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    # key may be None in cross-attention decode (already cached from encoder)
    num_kv_heads = key.shape[1] if key is not None else key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert value_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported FP8 kv_cache_dtype {kv_cache_dtype}: "
                f"should be one of 'fp8', 'fp8_e4m3', 'fp8_e5m2'."
            )

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    from vllm.platforms.rocm import use_rocm_custom_paged_attention

    use_custom = use_rocm_custom_paged_attention(
        query.dtype,
        head_size,
        block_size,
        num_queries_per_kv,
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        alibi_slopes,
        sinks,
    )
    has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)
    # Force Triton for non-standard blocks like Qwen3's 544 and for
    # stride-padded hybrid layouts. The latter use reshape_and_cache_flash
    # during cache update, so keep decode on the matching stride-aware path.
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
    if not is_pow2 or not has_native_layout:
        use_custom = False

    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions, head_size),
            dtype=query.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale=sm_scale,
            block_tables=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_out_scale=output_scale,
        )
    else:
        logger.warning_once(
            "Cannot use ROCm custom paged attention kernel,"
            " falling back to Triton implementation."
        )
        real_block_size = value_cache.shape[3]
        # The standard model directly uses the original block_size.
        # Non-standard 544 uses 32 to accommodate integer division logic.
        # Cap at 128 to avoid exceeding GPU shared memory limits
        # (e.g. hybrid Mamba models inflate block_size to 2048).
        # The kernel handles TRITON_BLOCK_SIZE != PHYSICAL_BLOCK_SIZE
        # via the l_block_idx/internal_offsets addressing logic.
        MAX_TRITON_BLOCK_SIZE = 128
        TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32
        if is_block_table_ptr:
            # Using the physical base address of tensors
            kv_element_size = key_cache.element_size()
            block_byte_stride = key_cache.stride(0) * kv_element_size
            # Get the starting physical address of the KV Cache
            base_addr = key_cache.data_ptr()

            # Normalization: Directly calculate the block offset
            # of the pointer relative to the base address
            processed_block_table = ((block_table - base_addr) // block_byte_stride).to(
                torch.int32
            )
        else:
            processed_block_table = block_table.to(torch.int32)

        kernel_paged_attention_2d[
            (
                num_seqs,
                num_kv_heads,
            )
        ](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=sinks,
            block_tables_ptr=processed_block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=processed_block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=TRITON_BLOCK_SIZE,
            PHYSICAL_BLOCK_SIZE=real_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=sinks is not None,
            USE_FP8=output_scale is not None,
        )




def _try_mixed_decode_prefill_split(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes,
    sliding_window,
    sm_scale,
    sinks,
    is_block_table_ptr,
    causal,
    output_scale,
) -> bool:
    """A-prime-prime-prime: split a batch that mixes short-query rows
    (decode / MTP verify, qlen <= 8) with long-query rows (prefill
    chunks, qlen > 8).

    Stock behavior: the batch-wide max_query_len trips the flash-decode
    gate, so the whole batch runs the standard context kernel -- and
    the decode requests with long cached contexts ride at cliff speed
    (~0.5 t/s measured) for the duration of the prefill step.

    Here: each uniform-qlen decode group runs the K-split flash-decode
    path (A'/A''); the prefill requests run the stock standard path on
    their own slice with a rebased query_start_loc. Together they cover
    every row exactly once.

    Cost: one small GPU->CPU sync (query_start_loc.tolist()) plus small
    row copies, paid only on the mixed step (a rare, already
    cliff-bound event). The v1 attention op runs eager (outside
    piecewise cudagraphs), so the sync is safe.

    Returns True if the split handled the batch (output fully written),
    False if the caller must run the standard path for the whole batch.
    """
    # --- common shape gate: identical to _try_flash_decode's ---
    if max_query_len <= _FLASH_DECODE_MAX_LQ:
        # Pure decode/verify batch: uniform -> _try_flash_decode (the
        # caller) already took it; non-uniform -> the standard path,
        # as in stock. Bailing out here keeps the per-step D2H sync
        # (query_start_loc.tolist() below) off the hot decode path.
        return False
    if output_scale is not None:
        return False
    if not causal:
        return False
    if alibi_slopes is not None or sinks is not None:
        return False
    if sliding_window and sliding_window > 0:
        return False
    if query.dtype not in (torch.half, torch.bfloat16):
        return False
    head_size = query.shape[2]
    if head_size > 256:
        return False
    if key_cache.ndim != 5 or value_cache.ndim != 4:
        return False
    real_block_size = value_cache.shape[3]
    if real_block_size < 2 or key_cache.shape[3] != real_block_size:
        return False
    if key_cache.shape[1] != value_cache.shape[1]:
        return False
    if key_cache.shape[2] * key_cache.shape[4] != head_size:
        return False
    if value_cache.shape[2] != head_size:
        return False

    # --- per-request query lengths (the one D2H sync of this path) ---
    qlens = query_start_loc.tolist()  # (num_seqs + 1,) cumulated
    num_seqs = len(qlens) - 1
    num_tokens = query.shape[0]
    if num_seqs == 0 or qlens[-1] != num_tokens:
        return False
    qlens_per_req = [qlens[i + 1] - qlens[i] for i in range(num_seqs)]
    decode_reqs = [
        i for i, l in enumerate(qlens_per_req)
        if 1 <= l <= _FLASH_DECODE_MAX_LQ
    ]
    prefill_reqs = [
        i for i, l in enumerate(qlens_per_req)
        if l > _FLASH_DECODE_MAX_LQ
    ]
    if not decode_reqs:
        return False
    # Split only when the standard path would slow down the decode rows:
    # a prefill request coexists, or the decode batch is non-uniform in
    # qlen (mixed MTP phases, e.g. lq=1 next to lq=4).
    decode_groups = {}
    for r in decode_reqs:
        decode_groups.setdefault(qlens_per_req[r], []).append(r)
    if not prefill_reqs and len(decode_groups) < 2:
        return False

    dev = query.device

    # --- decode groups: flash-decode path (uniform qlen per group) ---
    # NOTE: query/output are indexed by token rows; block_table/seq_lens
    # by REQUEST index. Two distinct index tensors -- mixing them is an
    # OOB (request idx > num_seqs faults the GPU on gfx1100).
    for lq in sorted(decode_groups):
        reqs = decode_groups[lq]
        rows = [qlens[i] + j for i in reqs for j in range(qlens_per_req[i])]
        idx = torch.tensor(rows, dtype=torch.int64, device=dev)
        idx_reqs = torch.tensor(reqs, dtype=torch.int64, device=dev)
        qg = query.index_select(0, idx).contiguous()
        og = torch.empty_like(qg)
        btg = block_table.index_select(0, idx_reqs).contiguous()
        slg = seq_lens.index_select(0, idx_reqs).contiguous()
        ok = _try_flash_decode(
            query=qg,
            output=og,
            kv_cache_dtype=kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=btg,
            seq_lens=slg,
            max_seq_len=max_seq_len,
            max_query_len=lq,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sinks=sinks,
            is_block_table_ptr=is_block_table_ptr,
            causal=causal,
            output_scale=output_scale,
        )
        if not ok:
            # Cannot happen by construction (uniform group + common
            # gate); fail loud rather than leave the row unwritten.
            raise RuntimeError(
                "A''' split: flash gate rejected a uniform decode group "
                f"(lq={lq}); partial output state"
            )
        output[idx] = og

    # --- prefill subset: stock standard path on its own slice ---
    if prefill_reqs:
        rows_p = [
            qlens[i] + j for i in prefill_reqs
            for j in range(qlens_per_req[i])
        ]
        idx_p = torch.tensor(rows_p, dtype=torch.int64, device=dev)
        idx_reqs_p = torch.tensor(prefill_reqs, dtype=torch.int64, device=dev)
        qp = query.index_select(0, idx_p).contiguous()
        op = output.index_select(0, idx_p).contiguous()
        btp = block_table.index_select(0, idx_reqs_p).contiguous()
        slp = seq_lens.index_select(0, idx_reqs_p).contiguous()
        # Rebased query_start_loc (cumulated within the subset): the
        # standard kernels index per-request rows through it.
        qslp = [0]
        for i in prefill_reqs:
            qslp.append(qslp[-1] + qlens_per_req[i])
        qslp_t = torch.tensor(
            qslp, dtype=query_start_loc.dtype, device=dev
        )
        kp = (key.index_select(0, idx_p).contiguous()
              if key is not None else None)
        vp = (value.index_select(0, idx_p).contiguous()
              if value is not None else None)
        max_q_pre = max(qlens_per_req[i] for i in prefill_reqs)
        _standard_context_decode_path(
            query=qp,
            key=kp,
            value=vp,
            output=op,
            kv_cache_dtype=kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=btp,
            query_start_loc=qslp_t,
            seq_lens=slp,
            max_seq_len=max_seq_len,
            max_query_len=max_q_pre,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            output_scale=output_scale,
            sinks=sinks,
            is_block_table_ptr=is_block_table_ptr,
            causal=causal,
        )
        output[idx_p] = op

    return True
