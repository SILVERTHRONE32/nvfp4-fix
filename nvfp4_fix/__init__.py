"""NVFP4 Fix - Patch compressed-tensors to support NVFP4 weight_scale buffers."""

__version__ = "0.1.0"

from .patches.patcher import apply_patch, is_patched
from .scripts.fix_model import fix_nvfp4_model

__all__ = ["apply_patch", "is_patched", "fix_nvfp4_model"]
