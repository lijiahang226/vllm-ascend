#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/worker.py
#

from __future__ import annotations

import functools
import math
import os
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch_npu  # noqa: F401
from packaging.version import InvalidVersion, Version
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import WeightPrefetchConfig, get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

COMPILATION_PASS_KEY = "graph_fusion_manager"
ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]
REGISTERED_ASCEND_OPS = {}


def check_nan(tensor: torch.Tensor, name: str, layer_name: str = "", extra_info: str = ""):
    """Check if tensor contains NaN values and print detailed information.
    
    Args:
        tensor: The tensor to check
        name: Name of the tensor (e.g., "hidden_states", "output")
        layer_name: Name of the layer (e.g., "model.layers.0")
        extra_info: Additional information to print
    """
    # Store original dtype
    original_dtype = tensor.dtype
    
    # Convert float8 types to bfloat16 for checking
    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]:
        tensor = tensor.to(torch.bfloat16)
    
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        print(f"\n{'='*80}")
        print(f"[NaN Detected] {name} in {layer_name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Original dtype: {original_dtype}")
        print(f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        
        if nan_count < total:
            valid_tensor = tensor[~torch.isnan(tensor)]
            print(f"  Min: {valid_tensor.min().item():.6e}")
            print(f"  Max: {valid_tensor.max().item():.6e}")
            print(f"  Mean: {valid_tensor.mean().item():.6e}")
            print(f"  Std: {valid_tensor.std().item():.6e}")
        else:
            print(f"  Min/Max/Mean/Std: All values are NaN!")
        
        if extra_info:
            print(f"  Extra info: {extra_info}")
        
        import traceback
        print("  Call stack:")
        for line in traceback.format_stack()[-6:-1]:
            print(f"    {line.strip()}")
        print(f"{'='*80}\n")
        return True
    return False
