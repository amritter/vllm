# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops  # type: ignore[no-redef]


class PagedAttention:
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1]
        # gfx1100 fork: V uses the SAME x-packed view as K --
        # (nb, Hkv, head_size//x, PAGE, x) instead of the upstream
        # (nb, Hkv, head_size, PAGE). The decode kernels stream KV in
        # (dim, token) tiles: with the upstream V layout each tile row
        # coalesces only BLOCK_N bytes of a 64 B cache line (~4x
        # over-fetch at BLOCK_N=16; the 2x100k profile showed the
        # flash-decode stage1 at ~135-180 GB/s effective of the 864 GB/s
        # roofline). X-packed V makes each (dim, token) tile row a 256 B
        # contiguous run (x=16: 16 tokens x 16 B), 1x over-fetch, with a
        # large row stride -- the exact access pattern K already uses and
        # that this kernel/codegen executes well. (Token-major V with
        # adjacent 256 B rows was tested and is ~16x slower in the real
        # kernel despite being identical in isolation -- the adjacent-row
        # pattern is pathological in the K+V+softmax context on this
        # driver; see docs/vllm-fork.md.) The x-pack formula degenerates
        # to the legacy (nb, Hkv, head_size, PAGE) layout for x=1, so no
        # layout flag is needed in the readers; the Triton writer writes
        # V with the same x-pack addressing as K.
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
