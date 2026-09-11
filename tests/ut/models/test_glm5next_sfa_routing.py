# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.config as vllm_config_module

import vllm_ascend.platform as platform
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, DeviceAdaptorFamily

# Routing and visible-length checks do not execute the external kernels.
with patch.dict(
    "sys.modules",
    {
        "vllm_ascend.ops.triton.glm5_next_kpool_state_compress": MagicMock(),
        "vllm_ascend.ops.triton.glm5_next_lightning_indexer": MagicMock(),
    },
):
    from vllm_ascend.models.glm5next.indexer import Glm5NextKPoolIndexerBackend


@pytest.mark.parametrize("kpool", [False, True])
@pytest.mark.parametrize("fp8_device", [False, True])
def test_sparse_models_select_shared_sfa(monkeypatch, kpool, fp8_device):
    config = SimpleNamespace(index_topk=2048)
    if kpool:
        config.index_kpool = 4
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config)),
    )
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            attention_backend_family=AttentionBackendFamily.STANDARD,
            device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED if fp8_device else None,
        ),
    )
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *args: False)
    selector = SimpleNamespace(use_mla=True, use_sparse=True, use_pcp=False)
    assert platform.NPUPlatform.get_attn_backend_cls(None, selector) == "vllm_ascend.attention.sfa_v1.AscendSFABackend"


@pytest.mark.parametrize("mode", ["310p", "pcp", "dcp"])
def test_kpool_unsupported_routes_fail_explicitly(monkeypatch, mode):
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=4))),
    )
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY
            if mode == "310p"
            else AttentionBackendFamily.STANDARD
        ),
    )
    selector = SimpleNamespace(use_mla=True, use_sparse=True, use_pcp=mode == "pcp", use_dcp=mode == "dcp")
    with pytest.raises(NotImplementedError, match="requires Ascend|context parallelism"):
        platform.NPUPlatform.get_attn_backend_cls(None, selector)


def _indexer():
    backend = object.__new__(Glm5NextKPoolIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.topk_tokens, backend.index_kpool = 2048, 4
    return backend


def test_indexer_owns_pool_boundary_lengths():
    indexer = _indexer()
    assert indexer.topk_output_width == 2051
    torch.testing.assert_close(
        indexer.get_topk_lengths(torch.tensor([0, 2, 3, 2047, 2048, 2050, 4095])),
        torch.tensor([1, 3, 4, 2048, 2049, 2051, 2048]),
    )
