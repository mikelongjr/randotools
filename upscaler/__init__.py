"""
RealESRGAN Upscaler - A PyQt6-based image and video upscaling application.

Supports NVIDIA (CUDA) and AMD (ROCm) GPUs, with CPU fallback.
Optimized for Fedora 43.
"""

__version__ = "2.0.0"
__author__ = "mikelongjr"

# Compatibility shims for dependencies
try:
    import sys
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F
except ImportError:
    pass

try:
    import huggingface_hub as _hf
    from huggingface_hub import hf_hub_download as _hf_dl
    _hf.cached_download = _hf_dl
except ImportError:
    pass
