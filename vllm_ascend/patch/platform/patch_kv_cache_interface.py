# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch
import vllm.model_executor.layers.attention.mla_attention
import vllm.v1.kv_cache_interface
from typing_extensions import Self
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _get_c8_k_cache_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8


def _get_c8_k_scale_cache_dtype() -> torch.dtype:
    return torch.float32 if get_ascend_device_type() == AscendDeviceType.A5 else torch.float16


@dataclass(frozen=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLAAttentionSpec extended to support DSA models, with optional Sparse C8 support.

    When Sparse C8 is enabled, the KV cache tuple changes from
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: bfloat16)
    to
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: int8, kv_cache[3]: float16).

    The semantic meaning of each KV cache entry is as follows:
    1. kv_cache[0] stores kv_lora.
    2. kv_cache[1] stores k_rope.
    3. kv_cache[2] stores the key tensor from the indexer module.
    4. kv_cache[3] stores the key scale tensor from the indexer module,
       and exists only when Sparse C8 is enabled.

    The main changes are as follows:
    1. The key tensor from the indexer module stored in kv_cache[2] is
       converted from bf16 to int8 to reduce memory usage. It is then
       processed with int8 precision in Lightning_indexer computation
       to improve computational efficiency.
    2. The quantization scale of the key tensor in the indexer module
       must also be stored for the Lightning_indexer_quant operator,
       and is therefore saved in kv_cache[3].
    """

    sparse_head_dim: tuple[int, ...] | None = None
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_cache_dtype)
    c8_k_scale_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_scale_cache_dtype)

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads
            
            kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim
            
            # A5: qk_rope_head_dim == 0 means kv_lora and k_rope are merged
            if qk_rope_head_dim == 0:
                # A5: ckv (merged kv_lora + k_rope)
                # A5 sparse C8: ckv uses float8_e4m3fn, not bfloat16
                ckv_dtype = self.c8_k_cache_dtype if self.cache_sparse_c8 else self.dtype
                ckv_bytes = num_heads_per_page * kv_lora_rank * get_dtype_size(ckv_dtype)
                # qli_tensor
                qli_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
                # qli_scale (per token, so head_dim is 1)
                qli_scale_bytes = num_heads_per_page * 1 * get_dtype_size(self.c8_k_scale_cache_dtype)
                return ckv_bytes + qli_bytes + qli_scale_bytes
            else:
                # A3: separate kv_lora and k_rope
                k_pe_nope_bytes = num_heads_per_page * (kv_lora_rank + qk_rope_head_dim) * get_dtype_size(self.dtype)
                indexer_k_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
                index_scale_head_dim = 1
                indexer_k_scale_bytes = (
                    num_heads_per_page * index_scale_head_dim * get_dtype_size(self.c8_k_scale_cache_dtype)
                )
                return k_pe_nope_bytes + indexer_k_bytes + indexer_k_scale_bytes

        return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @property
    def sparse_kv_cache_ratio(self) -> tuple[float, float, float, float | None]:
        """
        Compute the relative byte share of each KV cache entry.

        Returns:
            A tuple containing the ratios for:
            - kv_cache[0]
            - kv_cache[1]
            - kv_cache[2]
            - kv_cache[3] (None if Sparse C8 is disabled or A5 device)
        """

        assert self.sparse_head_dim is not None

        kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim

        if self.cache_sparse_c8:
            # A5: qk_rope_head_dim == 0 means kv_lora and k_rope are merged
            if qk_rope_head_dim == 0:
                # Calculate actual bytes for each tensor
                # A5 sparse C8: ckv uses float8_e4m3fn
                ckv_bytes = kv_lora_rank * get_dtype_size(self.c8_k_cache_dtype)
                qli_bytes = index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
                qli_scale_bytes = 1 * get_dtype_size(self.c8_k_scale_cache_dtype)
                total_bytes = ckv_bytes + qli_bytes + qli_scale_bytes

                return (
                    total_bytes / ckv_bytes,  # kv_cache[0]: ckv
                    total_bytes / qli_bytes,  # kv_cache[1]: qli_tensor
                    total_bytes / qli_scale_bytes,  # kv_cache[2]: qli_scale
                    None,  # kv_cache[3] does not exist for A5
                )
            else:
                # A3: separate kv_lora and k_rope
                k_bytes = kv_lora_rank * get_dtype_size(self.dtype)
                v_bytes = qk_rope_head_dim * get_dtype_size(self.dtype)
                qli_bytes = index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
                qli_scale_bytes = 1 * get_dtype_size(self.c8_k_scale_cache_dtype)
                total_bytes = k_bytes + v_bytes + qli_bytes + qli_scale_bytes

                return (
                    total_bytes / k_bytes,  # kv_cache[0]
                    total_bytes / v_bytes,  # kv_cache[1]
                    total_bytes / qli_bytes,  # kv_cache[2]
                    total_bytes / qli_scale_bytes,  # kv_cache[3]
                )

        return (
            self.head_size / self.sparse_head_dim[0],  # kv_cache[0]
            self.head_size / self.sparse_head_dim[1],  # kv_cache[1]
            self.head_size / self.sparse_head_dim[2],  # kv_cache[2]
            None,  # kv_cache[3] does not exist
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        assert len(cache_sparse_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse C8 setting."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            cache_sparse_c8=cache_sparse_c8_set.pop(),
        )


vllm.v1.kv_cache_interface.MLAAttentionSpec = AscendMLAAttentionSpec
vllm.model_executor.layers.attention.mla_attention.MLAAttentionSpec = AscendMLAAttentionSpec
