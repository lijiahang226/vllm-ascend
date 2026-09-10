# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("query_len", [1, 4, 6])
def test_dummy_uniform_decode_populates_each_speculative_width(query_len):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = object()
    runner.uniform_decode_query_len = query_len
    runner.num_decode_draft_tokens = SimpleNamespace(np=np.zeros(4, dtype=np.int32), copy_to_gpu=MagicMock())
    runner.num_accepted_tokens = SimpleNamespace(np=np.zeros(4, dtype=np.int32), copy_to_gpu=MagicMock())

    use_spec_decode = runner._prepare_dummy_spec_decode_metadata(
        np.array([query_len, query_len], dtype=np.int32), num_reqs=2, num_reqs_padded=4
    )

    assert use_spec_decode
    np.testing.assert_array_equal(runner.num_decode_draft_tokens.np, [query_len - 1] * 4)
    np.testing.assert_array_equal(runner.num_accepted_tokens.np, [query_len] * 4)
    runner.num_decode_draft_tokens.copy_to_gpu.assert_called_once_with()
    runner.num_accepted_tokens.copy_to_gpu.assert_called_once_with()
