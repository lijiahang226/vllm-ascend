# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import torch
import torch_npu

from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    QUANT_DTYPES,
    SCALE_DTYPES,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache, slot_indices=slot_mapping
        )

    @staticmethod
    def npu_moe_init_routing(
        hidden_states,
        topk_ids,
        *,
        scale=None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = -1,
    ):
        return torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
        return scale

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, out = torch.ops._C_ascend.moe_gating_top_k(
            x,
            k=k,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
            bias_opt=bias_opt,
        )
        return topk_weights, topk_ids.to(torch.int32), out

    @staticmethod
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        if dynamic_scale is None:
            return torch_npu.npu_dynamic_quant(hidden_states)

        return hidden_states, dynamic_scale

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return torch_npu.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=weight,
            bias=bias,
            group_list=group_list,
            weight_scale=weight_scale,
            x_scale=x_scale,
        )

    @staticmethod
    def get_quant_gmm2_kwargs(
        *,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
    ) -> dict:
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        return {
            "output_dtype": input_dtype if input_dtype in [torch.bfloat16, torch.float16] else torch.bfloat16,
        }

    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
        bias=None,
        fallback_output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if use_mxfp_quant:
            raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")

        if fallback_output_dtype is None:
            fallback_output_dtype = weight_scale[0].dtype if isinstance(weight_scale, list) else weight_scale.dtype
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=weight,
            scale=weight_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=fallback_output_dtype,
        )[0]

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
        torch_npu.atb.npu_paged_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts=seq_starts,
            key=key,
            value=value,
        )

    @staticmethod
    def mla_preprocess_only_decode(atten_obj, hidden_states, kv_cache, attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz]

        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], cos_shape[-1])

        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]
        dequant_scale_q_nope = None
        if atten_obj.fa_quant_layer:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope = torch_npu.npu_mla_prolog_v2(
                quantized_x,
                atten_obj.wd_q,
                atten_obj.wu_q,
                atten_obj.W_UK_T,
                atten_obj.wd_kv,
                atten_obj.gamma1,
                atten_obj.gamma2,
                sin,
                cos,
                attn_metadata.slot_mapping[:bsz].to(torch.int64),
                decode_k_nope,
                decode_k_pe,
                dequant_scale_x=pertoken_scale.view(-1, 1),
                dequant_scale_w_dq=atten_obj.dequant_scale_w_dq,
                dequant_scale_w_uq_qr=atten_obj.dequant_scale_w_uq_qr,
                dequant_scale_w_dkv_kr=atten_obj.dequant_scale_w_dkv_kr,
                quant_scale_ckv=atten_obj.quant_kscale,
                cache_mode="PA_NZ",
            )
        else:
            decode_q_nope = torch.empty(
                (hidden_states.shape[0], atten_obj.W_UK_T.shape[0], decode_k_nope.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            decode_q_pe = torch.empty(
                (hidden_states.shape[0], atten_obj.W_UK_T.shape[0], decode_k_pe.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            torch.ops._C_ascend.mla_preprocess(
                hidden_states,
                atten_obj.wd_qkv,
                atten_obj.deq_scale_qkv,
                atten_obj.gamma1,
                atten_obj.beta1,
                atten_obj.wu_q,
                atten_obj.qb_deq_scl,
                atten_obj.gamma2,
                cos,
                sin,
                atten_obj.W_UK_T,
                decode_k_nope,
                decode_k_pe,
                attn_metadata.slot_mapping[:bsz],
                quant_scale0=atten_obj.quant_scale0,
                quant_offset0=atten_obj.quant_offset0,
                bias0=atten_obj.quant_bias_qkv,
                quant_scale1=atten_obj.quant_scale1,
                quant_offset1=atten_obj.quant_offset1,
                bias1=atten_obj.qb_qt_bias,
                ctkv_scale=atten_obj.ctkv_scale,
                q_nope_scale=atten_obj.q_nope_scale,
                cache_mode="nzcache" if atten_obj.enable_kv_nz else "krope_ctkv",
                quant_mode="per_tensor_quant_asymm",
                q_out0=decode_q_nope,
                kv_cache_out0=decode_k_nope,
                q_out1=decode_q_pe,
                kv_cache_out1=decode_k_pe,
                enable_inner_out=False,
                inner_out=torch.tensor([], device=hidden_states.device),
            )
            decode_q_nope = decode_q_nope.view(bsz, atten_obj.num_heads, atten_obj.kv_lora_rank)
            decode_q_pe = decode_q_pe.view(bsz, atten_obj.num_heads, -1)

        decode_q_nope, decode_q_pe = atten_obj.reorg_decode_q(decode_q_nope, decode_q_pe)

        from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope=dequant_scale_q_nope
        )
        return decode_preprocess_res, None
    def sfa_preprocess_with_mlapo(
        sfa_impl,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
        num_actual_tokens: int,
    ) -> tuple:
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, sfa_impl.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, sfa_impl.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, sfa_impl.q_lora_rank),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            sfa_impl.wd_qkv,
            sfa_impl.deq_scale_qkv,
            sfa_impl.gamma1,
            sfa_impl.beta1,
            sfa_impl.wu_q,
            sfa_impl.qb_deq_scl,
            sfa_impl.gamma2,
            cos,
            sin,
            sfa_impl.W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            quant_scale0=sfa_impl.quant_scale0,
            quant_offset0=sfa_impl.quant_offset0,
            bias0=sfa_impl.quant_bias_qkv,
            quant_scale1=sfa_impl.quant_scale1,
            quant_offset1=sfa_impl.quant_offset1,
            bias1=sfa_impl.qb_qt_bias,
            ctkv_scale=sfa_impl.ctkv_scale,
            q_nope_scale=sfa_impl.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    @staticmethod
    def indexer_select_post_process(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        use_sparse_c8_indexer: bool,
        use_torch_npu_lightning_indexer: bool,
    ) -> torch.Tensor:
        if use_sparse_c8_indexer:
            assert len(kv_cache) == 4
            weights = weights.to(torch.float16)
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
                query=q_li.view(q_li_shape_ori),
                key=kv_cache[2],
                weights=weights,
                query_dequant_scale=q_li_scale.view(q_li_shape_ori[:-1]),
                key_dequant_scale=kv_cache[3].squeeze(2),
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        elif use_torch_npu_lightning_indexer:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices

    @staticmethod
    def execute_sparse_flash_attention_process(
        sfa_impl,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple,
        topk_indices: torch.Tensor,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        block_table = attn_metadata.block_table
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=sfa_impl.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        return attn_output

    def npu_flash_attention(query, key, value, seq_lens_cpu, head_num, scale_value, num_kv_heads):
        context_layer = torch.empty_like(query)

        torch_npu._npu_flash_attention_unpad(
            query=query,
            key=key,
            value=value,
            seq_len=seq_lens_cpu,
            scale_value=scale_value,
            num_heads=head_num,
            num_kv_heads=num_kv_heads,
            out=context_layer,
        )

        return context_layer


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
        )

    @staticmethod
    def npu_moe_init_routing(
        hidden_states,
        topk_ids,
        *,
        scale=None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = -1,
    ):
        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
        if scale is None or scale.ndim != 2:
            return scale
        if scale.shape[-1] % 2 != 0:
            raise ValueError(f"Invalid MXFP scale shape: {tuple(scale.shape)}")
        return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, out = torch_npu.npu_moe_gating_top_k(
            x,
            k=k,
            bias=bias_opt,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=0,
            norm_type=norm_type,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
        if norm_type == 0 and renorm == 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids.to(torch.int32), out

    @staticmethod
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_dynamic_quant(
                hidden_states,
                dynamic_scale,
                act_quant_type=act_quant_type,
                use_mxfp_quant=False,
            )

        if dynamic_scale is None:
            hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)

        return hidden_states, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(dynamic_scale)

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
        act_quant_type: torch.dtype | int = torch.float8_e4m3fn,
        weight_quant_type: torch.dtype | int = torch.float8_e4m3fn,
    ):
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant(
                x=x,
                weight=weight,
                group_list=group_list,
                weight_scale=weight_scale,
                x_scale=x_scale,
                bias=bias,
                use_mxfp_quant=False,
            )

        out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=x,
            weight=[weight],
            group_list=group_list,
            weight_scale=[weight_scale],
            x_scale=x_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=act_quant_type,
            x_dtype=act_quant_type if act_quant_type in QUANT_DTYPES else None,
            weight_dtype=weight_quant_type if weight_quant_type in QUANT_DTYPES else None,
            weight_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            x_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        )
        return out, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(out_scale), None

    @staticmethod
    def get_quant_gmm2_kwargs(
        *,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
    ) -> dict:
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.get_quant_gmm2_kwargs(
                input_dtype=input_dtype,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                scale_type=scale_type,
                per_token_scale_type=per_token_scale_type,
                use_bf16=use_bf16,
                use_mxfp_quant=False,
            )

        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )

        return {
            "scale_dtype": scale_type if scale_type in SCALE_DTYPES else None,
            "per_token_scale_dtype": per_token_scale_type if per_token_scale_type in SCALE_DTYPES else None,
            "x_dtype": act_quant_type if act_quant_type in QUANT_DTYPES else None,
            "weight_dtype": weight_quant_type if weight_quant_type in QUANT_DTYPES else None,
            "output_dtype": output_dtype,
        }

    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
        bias=None,
        fallback_output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_grouped_matmul_gmm2(
                hidden_states=hidden_states,
                weight=weight,
                weight_scale=weight_scale,
                per_token_scale=per_token_scale,
                group_list=group_list,
                group_list_type=group_list_type,
                input_dtype=input_dtype,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                scale_type=scale_type,
                per_token_scale_type=per_token_scale_type,
                use_bf16=use_bf16,
                use_mxfp_quant=False,
                bias=bias,
                fallback_output_dtype=fallback_output_dtype,
            )

        gmm2_kwargs = cls.get_quant_gmm2_kwargs(
            input_dtype=input_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=True,
        )
        output_dtype = gmm2_kwargs.pop("output_dtype")

        if isinstance(weight, list) and len(weight) != 1:
            raise ValueError(f"w2 must have a single tensor in MXFP path, but got {len(weight)}.")
        if isinstance(weight_scale, list) and len(weight_scale) != 1:
            raise ValueError(f"w2_scale must have a single tensor in MXFP path, but got {len(weight_scale)}.")
        gmm2_weight = weight if isinstance(weight, list) else [weight]
        gmm2_scale = weight_scale if isinstance(weight_scale, list) else [weight_scale]

        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=gmm2_weight,
            scale=gmm2_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
            **gmm2_kwargs,
        )[0]

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_offset, key, value):
        torch_npu.npu_gather_pa_kv_cache(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu.contiguous(),
            seq_offset=seq_offset,
            key=key,
            value=value,
        )

    @staticmethod
    def mla_preprocess_only_decode(atten_obj, hidden_states, kv_cache, attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz].unsqueeze(1)
        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        dynamic_scale = dynamic_scale.reshape(hidden_states.shape[0] * hidden_states.shape[1], -1)
        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], 1, cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], 1, cos_shape[-1])

        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]

        decode_q_nope, decode_q_pe, _, _, _ = torch_npu.npu_mla_prolog_v3(
            token_x=hidden_states,
            weight_dq=atten_obj.weight_dq,
            weight_uq_qr=atten_obj.weight_uq_qr,
            weight_uk=atten_obj.W_UK_T,
            weight_dkv_kr=atten_obj.weight_dkv_kr,
            rmsnorm_gamma_cq=atten_obj.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=atten_obj.kv_a_layernorm.weight.data,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=decode_k_nope,
            kr_cache=decode_k_pe,
            cache_index=attn_metadata.slot_mapping[:bsz].view(bsz, -1).to(torch.int64),
            dequant_scale_x=dynamic_scale.view(FLOAT8_E8M0FNU_DTYPE),
            dequant_scale_w_dq=atten_obj.weight_dq_scale.view(FLOAT8_E8M0FNU_DTYPE),
            dequant_scale_w_uq_qr=atten_obj.weight_uq_qr_scale.view(FLOAT8_E8M0FNU_DTYPE),
            dequant_scale_w_dkv_kr=atten_obj.weight_dkv_kr_scale.view(FLOAT8_E8M0FNU_DTYPE),
            cache_mode="PA_BSND",
            query_quant_mode=0,
            weight_quant_mode=3,
        )

        decode_q_nope = decode_q_nope.view(bsz, atten_obj.num_heads, atten_obj.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, atten_obj.num_heads, -1)

        decode_q_nope, decode_q_pe = atten_obj.reorg_decode_q(decode_q_nope, decode_q_pe)
        from vllm_ascend.attention.mla_v1 import DecodeMLAPreprocessResult

        decode_preprocess_res = DecodeMLAPreprocessResult(decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        return decode_preprocess_res, None
    def sfa_preprocess_with_mlapo(
        sfa_impl,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
        num_actual_tokens: int,
    ) -> tuple:
        total = hidden_states.shape[0]
        del num_input_tokens
        bsz = num_actual_tokens
        slot_mapping = slot_mapping[:bsz]
        hidden_states_temp = hidden_states[:bsz].unsqueeze(1)
        cos = cos[:bsz, ...]
        sin = sin[:bsz, ...]

        cos_shape = cos.shape
        cos = cos.view(cos_shape[0], 1, cos_shape[-1])
        sin = sin.view(cos_shape[0], 1, cos_shape[-1])

        decode_k_nope = kv_cache[0]
        use_c8 = getattr(sfa_impl, 'use_sparse_c8_indexer', False)
        kr_cache = torch.zeros(0, 0, decode_k_nope.shape[-2], cos_shape[-1], dtype=torch.bfloat16, device=decode_k_nope.device) if use_c8 else kv_cache[1]

        hidden_states_temp, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states_temp, dst_type=torch.float8_e4m3fn
        )
        dynamic_scale = dynamic_scale.reshape(hidden_states_temp.shape[0] * hidden_states_temp.shape[1], -1)

        decode_q_nope, q_pe, _, q_c, q_c_scale = torch_npu.npu_mla_prolog_v3(
            token_x=hidden_states_temp,
            weight_dq=sfa_impl.weight_dq,
            weight_uq_qr=sfa_impl.weight_uq_qr,
            weight_uk=sfa_impl.W_UK_T,
            weight_dkv_kr=sfa_impl.weight_dkv_kr,
            rmsnorm_gamma_cq=sfa_impl.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=sfa_impl.kv_a_layernorm.weight.data,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=decode_k_nope,
            kr_cache=kr_cache,
            cache_index=slot_mapping[:bsz].view(bsz, -1).to(torch.int64),
            dequant_scale_x=dynamic_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_dq=sfa_impl.weight_dq_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_uq_qr=sfa_impl.weight_uq_qr_scale.view(torch.float8_e8m0fnu),
            dequant_scale_w_dkv_kr=sfa_impl.weight_dkv_kr_scale.view(torch.float8_e8m0fnu),
            cache_mode="PA_BSND",
            weight_quant_mode=3,
            kv_cache_quant_mode=3 if use_c8 else 0,
            query_quant_mode=0,
            ckvkr_repo_mode=1 if use_c8 else 0,
            quant_scale_repo_mode=1 if use_c8 else 0,
            query_norm_flag=True
        )

        decode_q_nope = decode_q_nope.view(bsz, sfa_impl.num_heads, sfa_impl.kv_lora_rank)
        q_pe = q_pe.view(bsz, sfa_impl.num_heads, 64)

        if bsz < total:
            pad_size = total - bsz
            decode_q_nope = torch.nn.functional.pad(decode_q_nope, (0, 0, 0, 0, 0, pad_size))
            q_pe = torch.nn.functional.pad(q_pe, (0, 0, 0, 0, 0, pad_size))
            q_c = torch.nn.functional.pad(q_c, (0, 0, 0, pad_size))
            if q_c_scale is not None:
                q_c_scale = torch.nn.functional.pad(q_c_scale, (0, 0, 0, pad_size))

        return hidden_states, decode_q_nope, q_pe, (q_c, q_c_scale) if q_c_scale is not None else q_c

    @staticmethod
    def indexer_select_post_process(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple,
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        use_sparse_c8_indexer: bool,
        use_torch_npu_lightning_indexer: bool,
    ) -> torch.Tensor:
        if use_sparse_c8_indexer:
            assert len(kv_cache) == 3
            topk_indices = None

            q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])
            key_dequant_scale = kv_cache[2].squeeze(2)

            topk_indices = torch_npu.npu_quant_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=kv_cache[1],
                weights=weights,
                query_dequant_scale=q_li_scale,
                key_dequant_scale=key_dequant_scale,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li,
                key=kv_cache[2],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices

    @staticmethod
    def execute_sparse_flash_attention_process(
        sfa_impl,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple,
        topk_indices: torch.Tensor,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        block_table = attn_metadata.block_table
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        if kv.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            query = torch.cat([ql_nope, q_pe], dim=-1)

            attn_output = torch_npu.npu_kv_quant_sparse_flash_attention(
                query=query,
                key=kv,
                value=kv,
                sparse_indices=topk_indices,
                scale_value=sfa_impl.scale,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                layout_query="TND",
                layout_kv='PA_BSND',
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                key_quant_mode=2,
                value_quant_mode=2,
                rope_head_dim=64
            )
        else:
            attn_output, _, _ = torch_npu.npu_sparse_flash_attention(
                query=ql_nope,
                key=kv,
                value=kv,
                sparse_indices=topk_indices,
                scale_value=sfa_impl.scale,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                query_rope=q_pe,
                key_rope=key_rope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2
            )
        return attn_output

    def npu_flash_attention(query, key, value, seq_lens_cpu, head_num, scale_value, num_kv_heads):
        seq_lens_cpu = list(seq_lens_cpu.cumsum(0))

        context_layer = torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            actual_seq_qlen=seq_lens_cpu,
            actual_seq_kvlen=seq_lens_cpu,
            head_num=head_num,
            scale=scale_value,
            input_layout="TND",
        )[0]

        return context_layer


def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
