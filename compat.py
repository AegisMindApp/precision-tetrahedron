"""
Compatibility shims for PyTorch version differences.

  torch.autocast  — added in PyTorch 1.10; not available on local 1.8.x dev machine.
                    TPU VMs ship with PyTorch/XLA 2.x where it is always available.

Usage:
  from compat import autocast
  with autocast(device):
      out = model(x)
"""

import contextlib
import torch

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

_TORCH_AUTOCAST_AVAILABLE = hasattr(torch, "autocast")


@contextlib.contextmanager
def autocast(device=None, dtype=torch.bfloat16, enabled=True):
    """
    Unified autocast context:
      - XLA device  → torch.autocast("xla",  dtype=dtype)
      - CUDA device → torch.autocast("cuda", dtype=dtype)
      - CPU/old PT  → no-op (BF16 not useful on CPU anyway)
    """
    if not enabled or not _TORCH_AUTOCAST_AVAILABLE:
        yield
        return

    if XLA_AVAILABLE:
        device_type = "xla"
    elif device is not None and str(device).startswith("cuda"):
        device_type = "cuda"
    else:
        device_type = "cpu"

    with torch.autocast(device_type, dtype=dtype):
        yield
