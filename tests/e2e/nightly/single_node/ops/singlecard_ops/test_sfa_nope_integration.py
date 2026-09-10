# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU accuracy and graph replay coverage for GLM-5 sparse latent MLA."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded  # noqa: E402
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl  # noqa: E402
from vllm_ascend.attention.sparse_mla import SparseMLAMetadataState  # noqa: E402

NUM_QUERY_HEADS = 4
NUM_KV_HEADS = 1
LATENT_DIM = 512
ROPE_DIM = 64
BLOCK_SIZE = 128
NUM_BLOCKS = 4
SPARSE_WIDTH = 2048 + 3
SCALE = LATENT_DIM**-0.5
BF16_ATOL = 5e-2
BF16_RTOL = 2e-2


@pytest.fixture(scope="module")
def sparse_attention_op():
    ensure_c_ascend_loaded(required_op="npu_sparse_flash_attention")
    assert hasattr(torch.ops._C_ascend, "npu_sparse_flash_attention")
    return torch.ops._C_ascend.npu_sparse_flash_attention


def _make_cpu_case(
    seed: int,
    query_lens: tuple[int, int],
    seq_lens: tuple[int, int],
    block_table: list[list[int]],
    selected_tokens: list[list[int]],
    block_size: int = BLOCK_SIZE,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_tokens = sum(query_lens)
    query = (
        torch.randn(
            num_tokens,
            NUM_QUERY_HEADS,
            LATENT_DIM,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.1)
        .to(torch.bfloat16)
    )
    kv = torch.randn(
        NUM_BLOCKS,
        block_size,
        NUM_KV_HEADS,
        LATENT_DIM,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.1)
    for physical_block in range(NUM_BLOCKS):
        kv[physical_block].add_((physical_block + 1) * 0.125)
    kv = kv.to(torch.bfloat16)

    sparse_indices = torch.full(
        (num_tokens, NUM_KV_HEADS, SPARSE_WIDTH),
        -1,
        dtype=torch.int32,
    )
    for token_idx, indices in enumerate(selected_tokens):
        sparse_indices[token_idx, 0, : len(indices)] = torch.tensor(indices, dtype=torch.int32)

    query_ends = torch.tensor(query_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    table = torch.tensor(block_table, dtype=torch.int32)
    _validate_case(query_ends, seq_lens_tensor, table, sparse_indices)
    return {
        "query": query,
        "kv": kv,
        "sparse_indices": sparse_indices,
        "block_table": table,
        "actual_seq_lengths_query": query_ends,
        "actual_seq_lengths_kv": seq_lens_tensor,
        "query_rope": torch.zeros(num_tokens, NUM_QUERY_HEADS, ROPE_DIM, dtype=torch.bfloat16),
        "key_rope": torch.zeros(NUM_BLOCKS, block_size, NUM_KV_HEADS, ROPE_DIM, dtype=torch.bfloat16),
    }


def _validate_case(
    query_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> None:
    assert query_ends.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert sparse_indices.dtype == torch.int32
    assert sparse_indices.shape[-1] == SPARSE_WIDTH
    assert query_ends.tolist()[-1] == sparse_indices.shape[0]
    assert block_table.shape == (2, 2)

    query_begin = 0
    for request_idx, query_end in enumerate(query_ends.tolist()):
        query_len = query_end - query_begin
        seq_len = int(seq_lens[request_idx])
        for global_query_idx in range(query_begin, query_end):
            local_query_idx = global_query_idx - query_begin
            causal_limit = seq_len - query_len + local_query_idx + 1
            row = sparse_indices[global_query_idx, 0]
            valid = row[row >= 0]
            assert valid.numel() > 0
            assert bool((row[: valid.numel()] >= 0).all())
            assert bool((row[valid.numel() :] == -1).all())
            assert bool((valid < causal_limit).all())
        query_begin = query_end


def _to_npu(cpu_case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.to("npu") for name, tensor in cpu_case.items()}


def _copy_case_(npu_case: dict[str, torch.Tensor], cpu_case: dict[str, torch.Tensor]) -> None:
    for name, source in cpu_case.items():
        npu_case[name].copy_(source.to("npu"))


def _run_sparse_attention(op, inputs: dict[str, torch.Tensor]):
    return op(
        query=inputs["query"],
        key=inputs["kv"],
        value=inputs["kv"],
        sparse_indices=inputs["sparse_indices"],
        scale_value=SCALE,
        sparse_block_size=1,
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        query_rope=None,
        key_rope=None,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )


def _reference_attention(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    query = inputs["query"].float()
    kv = inputs["kv"].float()
    sparse_indices = inputs["sparse_indices"]
    block_table = inputs["block_table"].long()
    query_ends = inputs["actual_seq_lengths_query"].tolist()
    seq_lens = inputs["actual_seq_lengths_kv"].tolist()
    output = torch.empty_like(query)

    query_begin = 0
    for request_idx, query_end in enumerate(query_ends):
        query_len = query_end - query_begin
        seq_len = seq_lens[request_idx]
        for global_query_idx in range(query_begin, query_end):
            local_query_idx = global_query_idx - query_begin
            causal_limit = seq_len - query_len + local_query_idx + 1
            row = sparse_indices[global_query_idx, 0]
            valid = row[(row >= 0) & (row < causal_limit)].long()
            logical_blocks = torch.div(valid, kv.shape[1], rounding_mode="floor")
            block_offsets = torch.remainder(valid, kv.shape[1])
            physical_blocks = block_table[request_idx, logical_blocks]
            selected_kv = kv[physical_blocks, block_offsets, 0]
            scores = torch.matmul(query[global_query_idx], selected_kv.T) * SCALE
            probabilities = torch.softmax(scores, dim=-1)
            output[global_query_idx] = torch.matmul(probabilities, selected_kv)
        query_begin = query_end
    return output


def _assert_matches_reference(output: torch.Tensor, cpu_case: dict[str, torch.Tensor]) -> None:
    actual = output.cpu().float()
    expected = _reference_attention(cpu_case)
    assert output.shape == cpu_case["query"].shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=BF16_ATOL, rtol=BF16_RTOL)


def _first_case() -> dict[str, torch.Tensor]:
    return _make_cpu_case(
        seed=17,
        query_lens=(2, 1),
        seq_lens=(130, 65),
        block_table=[[2, 0], [3, 1]],
        selected_tokens=[[0, 17, 128], [1, 64, 129], [0, 32, 64]],
    )


def _replay_case() -> dict[str, torch.Tensor]:
    return _make_cpu_case(
        seed=29,
        query_lens=(1, 2),
        seq_lens=(64, 131),
        block_table=[[1, 3], [0, 2]],
        selected_tokens=[[2, 31, 63], [0, 64, 129], [3, 65, 130]],
    )


@torch.inference_mode()
def test_glm5next_nope_sparse_latent_attention_matches_reference(sparse_attention_op) -> None:
    cpu_case = _first_case()
    inputs = _to_npu(cpu_case)

    output, softmax_max, softmax_sum = _run_sparse_attention(sparse_attention_op, inputs)
    torch_npu.npu.synchronize()

    assert softmax_max.numel() == 0
    assert softmax_sum.numel() == 0
    _assert_matches_reference(output, cpu_case)


@pytest.mark.parametrize("page_padding_bytes", [0, 95232])
@torch.inference_mode()
def test_glm5next_sparse_mla_restores_c384_physical_pages(sparse_attention_op, page_padding_bytes) -> None:
    block_size = 384
    cpu_case = _make_cpu_case(
        seed=41,
        query_lens=(2, 1),
        seq_lens=(386, 193),
        block_table=[[2, 0], [3, 1]],
        selected_tokens=[[0, 127, 128, 383, 384], [1, 255, 385], [0, 128, 192]],
        block_size=block_size,
    )
    inputs = _to_npu(cpu_case)
    if page_padding_bytes:
        elements_per_page = block_size * NUM_KV_HEADS * LATENT_DIM
        padded = torch.full(
            (NUM_BLOCKS, elements_per_page + page_padding_bytes // 2),
            7.0,
            dtype=torch.bfloat16,
            device="npu",
        )
        cache = padded[:, :elements_per_page].view_as(inputs["kv"])
        cache.copy_(inputs["kv"])
        inputs["kv"] = cache
        assert not cache.is_contiguous()
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=block_size * 2),
    )
    builder = SparseMLAMetadataState(SimpleNamespace(block_size=block_size), config, torch.device("npu"), None)
    query_start_cpu = torch.tensor([0, 2, 3], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=386,
        query_start_loc=query_start_cpu.npu(),
        query_start_loc_cpu=query_start_cpu,
        seq_lens=inputs["actual_seq_lengths_kv"],
        block_table_tensor=(inputs["block_table"].unsqueeze(-1) * 3 + torch.arange(3, device="npu")).reshape(2, -1),
        slot_mapping=torch.tensor([384, 385, 3 * block_size + 192], device="npu"),
        positions=torch.tensor([384, 385, 192], device="npu"),
    )
    metadata = builder.prepare(SimpleNamespace(block_table=common.block_table_tensor))
    inputs["block_table"] = metadata.block_table
    output, _, _ = _run_sparse_attention(sparse_attention_op, inputs)
    torch_npu.npu.synchronize()
    _assert_matches_reference(output, cpu_case)


@pytest.mark.parametrize("page_padding_bytes", [0, 95232])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_glm5next_nope_cache_padding_is_a_noop(page_padding_bytes, use_graph):
    block_size = 384
    page_elements = block_size * LATENT_DIM
    backing = torch.full((2, page_elements + page_padding_bytes // 2), 3.0, dtype=torch.bfloat16, device="npu")
    cache = backing[:, :page_elements].view(2, block_size, 1, LATENT_DIM)
    values = torch.arange(4 * LATENT_DIM, dtype=torch.float32, device="npu").reshape(4, 1, 1, LATENT_DIM)
    values = (values / 512).bfloat16()
    slots = torch.tensor([0, block_size + 2, -1, -1], device="npu")
    impl = SimpleNamespace(qk_rope_head_dim=0, kv_lora_rank=LATENT_DIM, kv_a_layernorm=torch.nn.Identity())

    def write_cache():
        AscendSFAImpl.exec_kv(impl, values, None, None, (cache,), slots, None)

    write_cache()
    torch_npu.npu.synchronize()
    graph = torch.npu.NPUGraph() if use_graph else None
    try:
        if graph is not None:
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                write_cache()
        for next_slots in ([0, block_size + 2, -1, -1], [block_size + 3, 0, -1, -1]):
            before = backing.cpu().clone()
            expected_cache = before[:, :page_elements].view_as(cache)
            values.mul_(2)
            slots.copy_(torch.tensor(next_slots, device="npu"))
            cpu_values = values.cpu()
            for row, slot in enumerate(next_slots):
                if slot >= 0:
                    expected_cache[slot // block_size, slot % block_size] = cpu_values[row, 0]
            if graph is not None:
                graph.replay()
            else:
                write_cache()
            torch_npu.npu.synchronize()
            torch.testing.assert_close(backing.cpu(), before, atol=0, rtol=0)
    finally:
        if graph is not None:
            graph.reset()
        torch_npu.npu.synchronize()


@torch.inference_mode()
def test_glm5next_nope_sparse_latent_attention_graph_replays_tensor_metadata(
    sparse_attention_op,
) -> None:
    initial_cpu = _first_case()
    replay_cpu = _replay_case()
    inputs = _to_npu(initial_cpu)

    # Initialize the ACLNN path before capture, matching production graph warmup.
    _run_sparse_attention(sparse_attention_op, inputs)
    torch_npu.npu.synchronize()

    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(
            graph,
            capture_error_mode="thread_local",
            auto_dispatch_capture=True,
        ):
            output, softmax_max, softmax_sum = _run_sparse_attention(sparse_attention_op, inputs)

        graph.replay()
        torch_npu.npu.synchronize()
        _assert_matches_reference(output, initial_cpu)
        initial_output = output.cpu().clone()

        _copy_case_(inputs, replay_cpu)
        graph.replay()
        torch_npu.npu.synchronize()

        assert softmax_max.numel() == 0
        assert softmax_sum.numel() == 0
        _assert_matches_reference(output, replay_cpu)
        assert not torch.allclose(output.cpu(), initial_output, atol=BF16_ATOL, rtol=BF16_RTOL)
    finally:
        graph.reset()
        torch_npu.npu.synchronize()
