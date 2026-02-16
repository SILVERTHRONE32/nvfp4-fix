"""Optional low-memory mode patch for NVFP4 inference on consumer GPUs."""

import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear, QuantizationStatus

_original_forward = None
_patched = False

def enable_low_memory_mode():
    """
    Enable low-memory mode for NVFP4 models.
    
    This prevents caching of decompressed weights, keeping memory usage low
    at the cost of slightly slower inference (decompresses each forward pass).
    
    Essential for running 15B+ NVFP4 models on 16GB GPUs.
    """
    global _original_forward, _patched
    
    if _patched:
        print("Low-memory mode already enabled")
        return
    
    _original_forward = CompressedLinear.forward
    
    def compressed_forward(self, input):
        """Modified forward that decompresses on-the-fly without caching"""
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            # Decompress temporarily without caching
            weight_data = self.compressor.decompress_module(self)
            output = torch.nn.functional.linear(input, weight_data, self.bias)
            del weight_data  # Explicit cleanup
            return output
        else:
            return _original_forward(self, input)
    
    CompressedLinear.forward = compressed_forward
    _patched = True
    print("✓ Low-memory mode enabled for NVFP4")

def disable_low_memory_mode():
    """Restore original behavior (cache decompressed weights)."""
    global _patched
    
    if not _patched or _original_forward is None:
        print("Low-memory mode not enabled")
        return
    
    CompressedLinear.forward = _original_forward
    _patched = False
    print("✓ Low-memory mode disabled")

def is_low_memory_enabled():
    """Check if low-memory mode is active."""
    return _patched
