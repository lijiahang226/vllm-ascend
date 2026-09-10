# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

import vllm_ascend.patch.platform.patch_speculative_config as speculative_patch


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text", "glm5_next_mtp", "qwen3"])
def test_only_glm_mtp_disables_shared_topk(monkeypatch, model_type):
    config = PretrainedConfig(architectures=["TestModel"], index_share_for_mtp_iteration=True)
    config.model_type = model_type
    monkeypatch.setattr(speculative_patch, "_orig_hf_config_override", lambda value: value)
    for _ in range(2):
        result = speculative_patch._normalize_legacy_qwen3_dspark_config(config)
        assert result is config
        assert result.index_share_for_mtp_iteration is (model_type == "qwen3")
