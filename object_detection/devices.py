from __future__ import annotations

from typing import Any

from .errors import ConfigurationError

SUPPORTED_DEVICES = {"auto", "cpu", "mps", "cuda"}


def select_device(requested: str, torch_module: Any | None = None) -> str:
    requested = requested.lower()
    if requested not in SUPPORTED_DEVICES:
        raise ConfigurationError(
            f"Unsupported device '{requested}'. Choose auto, cpu, mps, or cuda."
        )

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            if requested in {"cuda", "mps"}:
                raise ConfigurationError(
                    f"Device '{requested}' requires PyTorch, which is not installed."
                )
            return "cpu"

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise ConfigurationError("CUDA was requested but no CUDA device is available.")
    if requested == "mps" and not mps_available:
        raise ConfigurationError("MPS was requested but Apple Metal acceleration is unavailable.")
    return requested

