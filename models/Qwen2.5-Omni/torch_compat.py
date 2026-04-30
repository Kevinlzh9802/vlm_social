#!/usr/bin/env python3
"""Small helpers for catching unsupported CUDA/PyTorch combinations early."""

from __future__ import annotations

from typing import Iterable

import torch


def _supported_sm_arches(arch_list: Iterable[str]) -> list[str]:
    return sorted(arch for arch in arch_list if arch.startswith("sm_"))


def ensure_cuda_runtime_compatibility() -> None:
    """Raise a clear error when the current GPU arch is missing from this build.

    PyTorch can report ``torch.cuda.is_available() == True`` even when the wheel
    was not compiled for the GPU's compute capability. In that case model loading
    may succeed, but inference later fails with ``no kernel image is available``.
    """

    if not torch.cuda.is_available():
        return

    supported_arches = _supported_sm_arches(torch.cuda.get_arch_list())
    if not supported_arches:
        return

    major, minor = torch.cuda.get_device_capability()
    device_arch = f"sm_{major}{minor}"
    if device_arch in supported_arches:
        return

    device_name = torch.cuda.get_device_name()
    supported_arch_text = " ".join(supported_arches)
    raise RuntimeError(
        "Unsupported CUDA architecture for this PyTorch build. "
        f"GPU '{device_name}' reports compute capability {device_arch}, but the installed "
        f"PyTorch build only supports: {supported_arch_text}. "
        "For Blackwell GPUs such as RTX PRO 6000 / sm_120, rebuild the runtime with "
        "a CUDA 12.8+ PyTorch wheel (for example torch 2.7.1, torchvision 0.22.1, "
        "torchaudio 2.7.1 from the cu128 index), or run the job on a GPU whose "
        "architecture is already listed above."
    )
