try:
    import pypto  # type: ignore[import-untyped]
    _HAS_PYPTO = True
except ImportError:
    _HAS_PYPTO = False

from typing import Any

import torch
from vllm.logger import logger


def _get_pypto_indexer_prolog_fn() -> Any:
    if not _HAS_PYPTO:
        return None

    try:
        from lightning_indexer_prolog_quant_mxfp8_impl import lightning_indexer_prolog_quant as _fn

        return _fn
    except ImportError:
        pass

    return None


def has_pypto() -> bool:
    return _HAS_PYPTO and _get_pypto_indexer_prolog_fn() is not None


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


def _log_pypto_arg_dtypes(**named_tensors) -> None:
    _dtypes = ", ".join(
        f"{name}={t.dtype}" for name, t in named_tensors.items() if isinstance(t, torch.Tensor)
    )
    logger.info("pypto_indexer arg dtypes: %s", _dtypes)


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
    pypto_fn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape

    pg_cache_index = (slot_mapping // block_size) * (block_size * head_dim) + (slot_mapping % block_size)
    pg_cache_index = pg_cache_index.view(t, 1)

    pg_scale_cache_index = (slot_mapping // block_size) * block_size + (slot_mapping % block_size)
    pg_scale_cache_index = pg_scale_cache_index.view(t, 1)

    k_cache_input = k_cache.reshape(block_num, -1, n_kv, head_dim)
    k_scale_input = k_scale_cache.reshape(block_num, -1, n_kv, 1)

    q_fp8 = torch.empty((t * head_num, head_dim), device=x.device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((t * head_num, 1), device=x.device, dtype=torch.float32)
    weights = torch.empty((t, head_num), device=x.device, dtype=torch.bfloat16)

    _log_pypto_arg_dtypes(
        x=x, q_norm=q_c, q_norm_scale=q_c_scale,
        w_qb=w_qb, w_qb_scale=w_qb_scale,
        wk=wk, w_proj=w_proj, gamma_k=gamma_k,
        cos=cos, sin=sin, hadamard_q=hadamard_q, hadamard_k=hadamard_k,
        k_quant=k_cache_input, k_scale=k_scale_input,
        k_cache_index=pg_cache_index, k_scale_cache_index=pg_scale_cache_index,
        q_out=q_fp8, q_scale_out=q_scale,
        k_out=k_cache_input, k_scale_out=k_scale_input,
        w_out=weights,
    )

    pypto_fn(
        x, q_c, q_c_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k,
        cos, sin, hadamard_q, hadamard_k,
        k_cache_input, k_scale_input,
        pg_cache_index, pg_scale_cache_index,
        q_fp8, q_scale, k_cache_input, k_scale_input, weights,
    )

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
    pypto_fn = _get_pypto_indexer_prolog_fn()
    if pypto_fn is None:
        raise RuntimeError(
            "lightning_indexer_prolog_quant_mxfp8_impl module not found. "
            "Ensure the pypto indexer prolog kernel is compiled and the "
            "containing directory is in PYTHONPATH."
        )
    return _pg_adapter_forward(
        x, q_c, q_c_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k,
        cos, sin, hadamard_q, hadamard_k,
        k_cache, k_scale_cache, slot_mapping,
        pypto_fn,
    )
