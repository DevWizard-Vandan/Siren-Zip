"""INT8/FP16 Model Quantizer and .neura Container Serialization Engine."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.model.siren_video import SirenVideo
from src.utils.neura_format import (
    HEADER_SIZE,
    QuantizedTensor,
    deserialize_header,
    deserialize_payload,
    serialize_header,
    serialize_payload,
)


def quantize_tensor_int8(tensor: torch.Tensor, name: str) -> QuantizedTensor:
    """Apply symmetric min-max quantization to INT8."""
    tensor_cpu = tensor.detach().cpu().float()
    max_val = torch.max(torch.abs(tensor_cpu)).item()
    scale = max_val / 127.0 if max_val > 0 else 1.0

    q_tensor = torch.clamp(torch.round(tensor_cpu / scale), -128, 127).to(torch.int8)
    q_bytes = q_tensor.numpy().tobytes()

    return QuantizedTensor(
        name=name,
        shape=tuple(tensor.shape),
        scale=scale,
        zero_point=0,
        data=q_bytes,
    )


def dequantize_tensor(q_tensor: QuantizedTensor, device: torch.device) -> torch.Tensor:
    """Dequantize INT8 bytes back to float32 tensor."""
    array_int8 = np.frombuffer(q_tensor.data, dtype=np.int8).copy().reshape(q_tensor.shape)
    tensor_float = torch.from_numpy(array_int8).to(device=device, dtype=torch.float32) * q_tensor.scale
    return tensor_float


def quantize_model(model: nn.Module) -> List[QuantizedTensor]:
    """Quantize all model parameters to INT8."""
    quantized_list: List[QuantizedTensor] = []
    for name, param in model.named_parameters():
        q_t = quantize_tensor_int8(param, name)
        quantized_list.append(q_t)
    return quantized_list


def dequantize_state_dict(
    quantized_tensors: List[QuantizedTensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert list of QuantizedTensors into a standard PyTorch state_dict."""
    state_dict: Dict[str, torch.Tensor] = {}
    for q_t in quantized_tensors:
        state_dict[q_t.name] = dequantize_tensor(q_t, device)
    return state_dict


def pack_neura_container(
    model: SirenVideo,
    video_meta: Dict[str, Any],
    output_path: str,
) -> int:
    """Pack trained SirenVideo model and video metadata into proprietary .neura file.

    Args:
        model: Trained SirenVideo instance.
        video_meta: Dict with 'frame_count', 'fps', 'width', 'height'.
        output_path: Destination path for .neura file.

    Returns:
        total_bytes: File size in bytes.
    """
    quantized_tensors = quantize_model(model)
    payload_bytes = serialize_payload(quantized_tensors)

    header_bytes = serialize_header(
        frame_count=video_meta.get("frame_count", 1),
        fps=video_meta.get("fps", 24.0),
        width=video_meta.get("width", 1280),
        height=video_meta.get("height", 720),
        hidden_layers=model.hidden_layers,
        hidden_features=model.hidden_features,
        omega_xy=model.omega_xy,
        omega_t=model.omega_t,
        omega_0_hidden=model.omega_0_hidden,
        final_activation=model.final_activation,
        num_tensors=len(quantized_tensors),
        payload_size=len(payload_bytes),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header_bytes)
        f.write(payload_bytes)

    total_bytes = len(header_bytes) + len(payload_bytes)
    return total_bytes


def unpack_neura_container(
    neura_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[SirenVideo, Dict[str, Any]]:
    """Load and dequantize a .neura container into an executable SirenVideo model.

    Args:
        neura_path: Path to .neura file.
        device: Torch device.

    Returns:
        model: Reconstructed and dequantized SirenVideo model.
        meta: Video metadata dictionary.
    """
    torch_device = torch.device(device)
    with open(neura_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        meta = deserialize_header(header_bytes)

        payload_bytes = f.read(meta["payload_size"])

    quantized_tensors = deserialize_payload(payload_bytes, meta["num_tensors"])
    state_dict = dequantize_state_dict(quantized_tensors, torch_device)

    model = SirenVideo(
        in_features=3,
        hidden_features=meta["hidden_features"],
        hidden_layers=meta["hidden_layers"],
        out_features=3,
        omega_xy=meta["omega_xy"],
        omega_t=meta["omega_t"],
        omega_0_hidden=meta["omega_0_hidden"],
        final_activation=meta["final_activation"],
    )

    model.to(torch_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, meta
