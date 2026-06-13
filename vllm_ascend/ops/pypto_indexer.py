try:
    import pypto  # type: ignore[import-untyped]
    _HAS_PYPTO = True
except ImportError:
    _HAS_PYPTO = False

from typing import Any

import torch


def has_pypto() -> bool:
    return _HAS_PYPTO


def register_pypto_indexer_op():
    if not _HAS_PYPTO:
        return

    try:
        # Already registered?
        torch.ops.pypto.lightning_indexer_prolog_quant_mxfp8
        return
    except (AttributeError, RuntimeError):
        pass

    pyptolib = torch.library.Library("pypto", "FRAGMENT")
    pyptolib.define(
        "lightning_indexer_prolog_quant_mxfp8("
        "Tensor x, Tensor q_norm, Tensor q_norm_scale, "
        "Tensor w_qb, Tensor w_qb_scale, "
        "Tensor wk, Tensor w_proj, Tensor gamma_k, "
        "Tensor cos_idx_rope, Tensor sin_idx_rope, "
        "Tensor hadamard_q, Tensor hadamard_k, "
        "Tensor k_cache, Tensor k_scale_cache, "
        "Tensor k_cache_index, Tensor k_scale_cache_index"
        ") -> (Tensor q_fp8e4m3, Tensor q_scale, Tensor k_fp8e4m3, Tensor k_scale, Tensor weights)"
    )

    @torch.library.impl(pyptolib, "lightning_indexer_prolog_quant_mxfp8", "Meta")
    def _meta(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k,
              cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k,
              k_cache, k_scale_cache, k_cache_index, k_scale_cache_index):
        t = x.shape[0]
        head_num = w_proj.shape[1]
        block_num, block_size, n_kv, head_dim = k_cache.shape
        q_fp8 = torch.empty((t * head_num, head_dim), device="meta", dtype=torch.float8_e4m3fn)
        q_scale = torch.empty((t * head_num, 1), device="meta", dtype=torch.float32)
        weights = torch.empty((t, head_num), device="meta", dtype=torch.bfloat16)
        return q_fp8, q_scale, k_cache, k_scale_cache, weights


def _build_pg_buffer_from_contiguous_cache(
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert contiguous vllm-ascend kv cache tensors into a PG-compatible buffer.

    vllm-ascend stores k_cache (fp8) and k_scale_cache (fp32) as separate contiguous tensors
    with shape (block_num, block_size, n_kv, head_dim or 1). The pypto PG kernel expects
    them packed in a single uint8 buffer with interleaved layout.

    Returns:
        pg_buffer: uint8 tensor containing packed cache data
        k_cache_pg: PG-compatible strided view for k_cache
        k_scale_cache_pg: PG-compatible strided view for k_scale_cache
    """
    block_num, block_size, n_kv, hd = k_cache.shape
    d = head_dim

    fp8_data_per_block = block_size * n_kv * d       # fp8 elements
    fp8_offset = fp8_data_per_block                   # skip padding
    fp32_offset_bytes = block_size * n_kv * 2 * d     # skip to fp32 section
    fp32_vals = block_size * n_kv                     # fp32 elements per block

    total_uint8_per_block = fp32_offset_bytes + fp32_vals * 4 + fp8_offset
    pg_buffer = torch.zeros(block_num, total_uint8_per_block, dtype=torch.uint8, device=k_cache.device)

    # Copy k_cache into pg_buffer fp8 section
    k_section = pg_buffer.view(torch.float8_e4m3fn)[:, fp8_offset:fp8_offset + fp8_data_per_block]
    k_section_4d = k_section.view(block_num, block_size, n_kv, hd)
    k_section_4d.copy_(k_cache.view(block_num, block_size, n_kv, hd))

    # Copy k_scale_cache into pg_buffer fp32 section
    ks_start = fp32_offset_bytes // 4
    ks_end = (fp32_offset_bytes + fp32_vals * 4) // 4
    ks_section = pg_buffer.view(torch.float32)[:, ks_start:ks_end]
    ks_section_4d = ks_section.view(block_num, block_size, n_kv, 1)
    ks_section_4d.copy_(k_scale_cache.view(block_num, block_size, n_kv, 1))

    # PG page_size: offset to reach fp32 section, in units of (n_kv * block_size) elements
    page_size = fp32_offset_bytes // (n_kv * block_size)

    # Create strided PG-compatible views (non-contiguous by construction)
    k_cache_pg = pg_buffer.view(torch.float8_e4m3fn)
    k_cache_pg = torch.as_strided(
        k_cache_pg,
        size=(block_num, block_size, n_kv, page_size),
        stride=(block_size * n_kv * page_size, n_kv * page_size, page_size, 1),
    )
    k_scale_pg = pg_buffer.view(torch.float32)
    k_scale_pg = torch.as_strided(
        k_scale_pg,
        size=(block_num, block_size, n_kv, page_size // 4),
        stride=(block_size * n_kv * page_size // 4, n_kv * page_size // 4, page_size // 4, 1),
    )

    return pg_buffer, k_cache_pg, k_scale_pg


def _extract_from_pg_buffer(
    pg_buffer: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    head_dim: int,
) -> None:
    """Extract updated k_cache and k_scale from PG buffer back to vllm-ascend tensors."""
    block_num, block_size, n_kv, hd = k_cache.shape
    d = head_dim

    fp8_data_per_block = block_size * n_kv * d
    fp8_offset = fp8_data_per_block
    fp32_offset_bytes = block_size * n_kv * 2 * d
    fp32_vals = block_size * n_kv

    k_section = pg_buffer.view(torch.float8_e4m3fn)[:, fp8_offset:fp8_offset + fp8_data_per_block]
    k_cache.view(block_num, block_size, n_kv, hd).copy_(k_section.view(block_num, block_size, n_kv, hd))

    ks_start = fp32_offset_bytes // 4
    ks_end = (fp32_offset_bytes + fp32_vals * 4) // 4
    ks_section = pg_buffer.view(torch.float32)[:, ks_start:ks_end]
    k_scale_cache.view(block_num, block_size, n_kv, 1).copy_(ks_section.view(block_num, block_size, n_kv, 1))


def _compute_pg_cache_indices(
    cache_index: torch.Tensor,
    block_size: int,
    k_cache_shape_per_block: int,
    n_kv: int,
    storage_offset: int,
    element_size: int,
) -> torch.Tensor:
    """Compute PG-compatible cache indices from flat slot_mapping."""
    # Formula from pypto test: cache_index // block_size * (cache_dim) * block_size + cache_index % block_size + offset
    # For fp8: cache_dim = k_cache_shape[-1] // head_dim
    # For fp32 scale: similar with appropriate units
    t = cache_index.shape[0]
    block_idx = cache_index // block_size
    offset_in_block = cache_index % block_size
    pg_index = block_idx * k_cache_shape_per_block * block_size + offset_in_block + storage_offset // element_size
    return pg_index.view(t, 1)


def _pg_adapter_forward(
    x: torch.Tensor,
    q_c: torch.Tensor,
    q_c_scale: torch.Tensor,
    w_qb: torch.Tensor,
    w_qb_scale: torch.Tensor,
    wk: torch.Tensor,
    w_proj: torch.Tensor,
    gamma_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    hadamard_q: torch.Tensor,
    hadamard_k: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape

    # Build PG buffer and views
    pg_buffer, k_cache_pg, k_scale_pg = _build_pg_buffer_from_contiguous_cache(
        k_cache, k_scale_cache, head_dim,
    )

    # Compute PG-compatible indices (flat slot_mapping → PG index)
    k_storage_offset = block_size * n_kv * head_dim  # fp8 offset in bytes (element count = bytes for fp8)
    k_scale_storage_offset = block_size * n_kv * 2 * head_dim // 4  # fp32 offset

    # PG page_size to compute k_cache_shape_per_block
    page_size = block_size * n_kv * 2 * head_dim // (n_kv * block_size)  # = 2 * head_dim

    pg_cache_index = _compute_pg_cache_indices(
        slot_mapping, block_size, page_size // head_dim, n_kv, k_storage_offset, 1,
    )
    pg_scale_cache_index = _compute_pg_cache_indices(
        slot_mapping, block_size, page_size // 4, n_kv, k_scale_storage_offset * 4, 4,
    )

    # Reshape for pypto: k_cache shape (block_num, block_size * cache_dim, n_kv, head_dim)
    k_cache_input = k_cache_pg.view(block_num, block_size * (k_cache_pg.shape[-1] // head_dim), n_kv, head_dim)
    k_scale_input = k_scale_pg.view(block_num, block_size * (k_scale_pg.shape[-1] // 1), n_kv, 1)

    # Allocate output buffers
    q_fp8 = torch.empty((t * head_num, head_dim), device=x.device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((t * head_num, 1), device=x.device, dtype=torch.float32)
    weights = torch.empty((t, head_num), device=x.device, dtype=torch.bfloat16)

    from lightning_indexer_prolog_quant_mxfp8_impl import lightning_indexer_prolog_quant

    lightning_indexer_prolog_quant(
        x, q_c, q_c_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k,
        cos, sin, hadamard_q, hadamard_k,
        k_cache_input, k_scale_input,
        pg_cache_index, pg_scale_cache_index,
        q_fp8, q_scale, k_cache_input, k_scale_input, weights,
    )

    # Extract updated k_cache / k_scale back
    _extract_from_pg_buffer(pg_buffer, k_cache, k_scale_cache, head_dim)

    return q_fp8, q_scale, weights


def lightning_indexer_prolog_quant_mxfp8(
    x: torch.Tensor,
    q_c: torch.Tensor,
    q_c_scale: torch.Tensor,
    w_qb: torch.Tensor,
    w_qb_scale: torch.Tensor,
    wk: torch.Tensor,
    w_proj: torch.Tensor,
    gamma_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    hadamard_q: torch.Tensor,
    hadamard_k: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _pg_adapter_forward(
        x, q_c, q_c_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k,
        cos, sin, hadamard_q, hadamard_k,
        k_cache, k_scale_cache, slot_mapping,
    )
