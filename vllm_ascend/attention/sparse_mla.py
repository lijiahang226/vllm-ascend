# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NoPE sparse MLA contracts for A2/A3 and A5."""

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

SMLA_METADATA_SIZE = 1024
SPARSE_ATTENTION_MAX_BLOCK_SIZE = 1024


def build_smla_metadata(metadata, buffer, num_heads, head_dim, topk):
    generated = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        num_heads_q=num_heads,
        num_heads_kv=1,
        head_dim=head_dim,
        cu_seqlens_q=metadata.query_start_loc,
        seqused_ori_kv=metadata.seq_lens,
        ori_topk_length=metadata.smla_topk_length,
        batch_size=metadata.seq_lens.numel(),
        # These are scheduler CPU scalars. Passing device max() results here
        # would force a host synchronization for each metadata build.
        max_seqlen_q=metadata.max_query_len,
        max_seqlen_ori_kv=metadata.max_seq_len,
        max_seqlen_cmp_kv=0,
        ori_topk=topk,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=3,
        cmp_mask_mode=3,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(metadata.seq_lens.device),
    )
    buffer.copy_(generated)
    metadata.smla_metadata = buffer


def sparse_mla(query, cache, indices, metadata, scale):
    """Attend to original latent KV, using the platform's NoPE operator."""
    if metadata.smla_metadata is not None:
        # The A5 DMA merges adjacent columns. Preserve the selected set while
        # sorting token positions and moving invalid padding to the end.
        sentinel = torch.iinfo(torch.int32).max
        sorted_indices = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
        sorted_indices = torch.where(sorted_indices == sentinel, -1, sorted_indices)
        result = torch.ops._C_ascend.npu_sparse_flash_mla(
            query.contiguous(),
            ori_kv=cache,
            ori_sparse_indices=sorted_indices,
            ori_block_table=metadata.block_table,
            cu_seqlens_q=metadata.query_start_loc,
            seqused_ori_kv=metadata.seq_lens,
            ori_topk_length=metadata.smla_topk_length[: query.shape[0]],
            sinks=None,
            metadata=metadata.smla_metadata,
            softmax_scale=scale,
            cmp_ratio=1,
            ori_mask_mode=3,
            cmp_mask_mode=3,
            ori_win_left=0,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )
    else:
        # Large, contiguous hybrid storage pages need a C128 view to meet
        # the A2/A3 limit. Keep supported page-strided layouts unchanged.
        if cache.shape[1] != metadata.block_size:
            cache = cache.view(-1, metadata.block_size, *cache.shape[2:])
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query.contiguous(),
            key=cache,
            value=cache,
            sparse_indices=indices,
            scale_value=scale,
            sparse_block_size=1,
            block_table=metadata.block_table,
            actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32),
            actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
            query_rope=None,
            key_rope=None,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
    output = result[0]
    # Kernels may leave graph-capacity rows unwritten. Mask on device before
    # value/output projections so NaNs in padding cannot escape the layer.
    valid = torch.arange(query.shape[0], device=query.device) < metadata.query_start_loc[-1]
    return output.masked_fill(~valid[:, None, None], 0)


class SparseMLAMetadataState:
    """Persistent operator buffers for NoPE within the shared SFA builder.

    Indexers supply their visible index counts. Pool construction and scoring
    remain entirely outside attention metadata and operator dispatch.
    """

    def __init__(self, kv_cache_spec, vllm_config, device, indexer, kernel_block_size=128):
        block_size = kv_cache_spec.block_size
        if block_size <= 0 or block_size % kernel_block_size:
            raise ValueError("Sparse MLA block size must be a positive multiple of the SFA kernel block size.")
        self.split = block_size // kernel_block_size
        self.use_smla = get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        self.block_size = block_size
        if not self.use_smla and block_size > SPARSE_ATTENTION_MAX_BLOCK_SIZE:
            self.block_size = kernel_block_size
        self.table_stride = self.block_size // kernel_block_size
        table_width = cdiv(vllm_config.model_config.max_model_len, block_size) * (block_size // self.block_size)
        self.block_table_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            table_width,
            dtype=torch.int32,
            device=device,
        )
        self.indexer = indexer
        if self.use_smla:
            if indexer is None:
                raise ValueError("A5 NoPE sparse MLA requires an indexer to supply visible top-k lengths.")
            config = vllm_config.model_config.hf_text_config
            self.num_heads = config.num_attention_heads // vllm_config.parallel_config.tensor_parallel_size
            self.head_dim = config.kv_lora_rank
            self.metadata_buffer = torch.empty(SMLA_METADATA_SIZE, dtype=torch.int32, device=device)
            self.length_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                1,
                dtype=torch.int32,
                device=device,
            )

    def prepare(self, metadata):
        expanded = metadata.block_table
        if expanded.shape[1] % self.split:
            raise ValueError("Sparse MLA received a partially expanded SFA block table.")
        width = expanded.shape[1] // self.table_stride
        if width > self.block_table_buffer.shape[1]:
            raise ValueError("Sparse MLA block table exceeds its persistent buffer.")
        table = self.block_table_buffer[: expanded.shape[0], :width]
        torch.div(expanded[:, :: self.table_stride], self.table_stride, rounding_mode="floor", out=table)
        metadata.block_table = table
        metadata.block_size = self.block_size
        if self.use_smla:
            positions = metadata.positions
            if positions.numel() > self.length_buffer.shape[0]:
                raise ValueError("Sparse MLA token count exceeds its persistent top-k buffer.")
            lengths = self.length_buffer[: positions.numel()]
            counts = self.indexer.get_topk_lengths(positions)
            valid = torch.arange(positions.numel(), device=positions.device) < metadata.query_start_loc[-1]
            lengths[:, 0].copy_(counts.masked_fill(~valid, 0))
            metadata.smla_topk_length = lengths
            build_smla_metadata(
                metadata, self.metadata_buffer, self.num_heads, self.head_dim, self.indexer.topk_output_width
            )
        return metadata
