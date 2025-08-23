# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Any, Optional, Union

import numpy as np
import torch


class AdaQuantizer:
    def __init__(self, weight: float = 0.01) -> None:
        self.weight = weight

    def quantize(self, values_tensor: torch.Tensor) -> tuple[Union[torch.Tensor, np.ndarray], dict]:
        values_tensor = values_tensor.cpu()
        old_values_tensor = values_tensor

        old_tensor_shape = list(old_values_tensor.shape)
        values_tensor = values_tensor.to(dtype=torch.float64).view(-1)
        offset = self.get_offset(values_tensor)
        values_tensor = values_tensor + offset
        norm = values_tensor.abs().max()

        quant_state = {"offset": offset}
        res = self.get_number_of_quantization_levels(
            element_size=old_values_tensor.element_size(), values_tensor=values_tensor
        )
        if res is None:
            return old_values_tensor, {}
        norm, quantization_level, new_dtype = res
        if norm == 0.0:
            return torch.tensor([0], dtype=torch.bool), quant_state | {
                "tensor_shape": old_tensor_shape,
            }

        normalized_abs_tensor = values_tensor.abs() / norm
        quantized_tensor = (normalized_abs_tensor * quantization_level).round().clamp(0, quantization_level)
        quantized_tensor = quantized_tensor.numpy().astype(dtype=new_dtype)
        return quantized_tensor, quant_state | {
            "quantization_level": quantization_level,
            "tensor_shape": old_tensor_shape,
            "norm": norm,
        }

    def get_number_of_quantization_levels(
        self, element_size: int, values_tensor: torch.Tensor
    ) -> Optional[tuple[float, int, Any]]:
        norm = values_tensor.abs().max().item()
        element_bits = element_size * 8
        quantization_level = math.ceil(max(1, math.sqrt(norm * element_bits * math.log(4) / self.weight)))
        new_element_bits = math.ceil(math.log2(quantization_level))
        print("new element_bits is", new_element_bits)
        quantization_level = int(2**new_element_bits) - 1
        # print("quantization_level is", quantization_level)
        new_dtype = None
        if new_element_bits < element_bits:
            if new_element_bits <= 8:
                new_dtype = np.uint8
            elif new_element_bits <= 16:
                new_dtype = "<u2"
            else:
                raise RuntimeError(f"Invalid element_bits {new_element_bits}")
        if new_dtype is None:
            return None
        return norm, quantization_level, new_dtype

    def dequantized(self, quantized_tensor: torch.Tensor, quant_state: dict) -> torch.Tensor:
        offset = quant_state["offset"]
        if "norm" not in quant_state:
            return torch.zeros(quant_state["tensor_shape"], dtype=torch.float64) - offset
        norm = quant_state["norm"]
        quantization_level = quant_state["quantization_level"]
        quantized_tensor = quantized_tensor.to(dtype=torch.float64).reshape(quant_state["tensor_shape"])

        return (quantized_tensor * norm / quantization_level) - offset

    def get_offset(self, tensor: torch.Tensor) -> float:
        min_value = tensor.min().item()
        return -min_value
