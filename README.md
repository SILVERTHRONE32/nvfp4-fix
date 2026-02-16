# NVFP4 Fix for compressed-tensors

Fix `KeyError: 'weight_scale'` when loading NVFP4 quantized models.

## 🚀 Quick Start
```bash
pip install git+https://github.com/SILVERTHRONE32/nvfp4-fix.git
```

### For Fast Inference (Recommended - vLLM)
```python
import os
os.environ['VLLM_USE_V1'] = '0'

from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/nvfp4-model",
    quantization="compressed-tensors",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

outputs = llm.generate("Hello!", SamplingParams(max_tokens=50))
# Speed: ~23-27 tokens/sec with fused FP4 CUDA kernels!
```

### For Transformers (Slower)
```python
from nvfp4_fix import apply_patch, enable_low_memory_mode
from transformers import AutoModelForCausalLM

# Apply patches
apply_patch()
enable_low_memory_mode()  # For 16GB GPUs

model = AutoModelForCausalLM.from_pretrained("path/to/nvfp4-model")
# Speed: ~0.4 tokens/sec (but fits in 16GB)
```

## 🐛 What This Fixes

- ✅ `KeyError: 'weight_scale'` during NVFP4 model loading
- ✅ `RuntimeError: Promotion for Float8 Types is not supported`
- ✅ Enables inference on consumer 16GB GPUs
- ✅ Works with LLaVA, Mistral, Llama, Gemma, and other NVFP4 models

## 📖 The Problem

NVFP4 (NVIDIA FP4) quantized models fail because:

1. **weight_scale skipped during loading**: `_skip_scale()` returns `True` for NVFP4
2. **Buffers not included in decompression**: Only parameters are extracted
3. **Float8 dtype conflicts**: Scales stored as `float8_e4m3fn` conflict with `bfloat16`

## 🔧 How It Works

This package applies **three patches** to compressed-tensors:

### Patch 1: Force load weight_scale
```python
def _skip_scale(self):
    return False  # Don't skip weight_scale!
```

### Patch 2: Include buffers in decompression
```python
for name, buffer in module.named_buffers():
    compressed_data[name] = buffer
```

### Patch 3: Convert Float8 to BFloat16
```python
if scale.dtype == torch.float8_e4m3fn:
    scale = scale.to(torch.bfloat16)
```

## ⚡ Performance Comparison

| Method | Speed | Memory | Notes |
|--------|-------|--------|-------|
| **vLLM (V0)** | **23-27 tok/s** | ~15 GB | ✅ Fused FP4 kernels (`cutlass_scaled_fp4_mm`) |
| Transformers | 0.4 tok/s | ~15 GB | ❌ Decompresses on every forward pass |
| Transformers + low-memory | 0.4 tok/s | ~11 GB | ✅ Fits 16GB but slow |

**Recommendation**: Use vLLM for production. It's 50x faster!

## 📦 Installation
```bash
pip install git+https://github.com/SILVERTHRONE32/nvfp4-fix.git
```

## 🎯 Usage

### Apply Patches (One-time)
```bash
nvfp4-fix apply-patch
```

Or in Python:
```python
from nvfp4_fix import apply_patch
apply_patch()
```

### Check If Patched
```bash
nvfp4-fix check
```

### Low-Memory Mode (Optional - for 16GB GPUs with Transformers)
```python
from nvfp4_fix import enable_low_memory_mode

enable_low_memory_mode()
# Now load your model with transformers
```

This prevents caching decompressed weights, keeping memory low at the cost of speed.

## 🧪 Verification
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "path/to/nvfp4-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Check it's still quantized
q_proj = model.model.layers[0].self_attn.q_proj
assert hasattr(q_proj, 'weight_packed')
assert q_proj.weight_packed.dtype == torch.uint8
print("✓ Model is properly NVFP4 quantized!")
```

## 📤 For Model Creators

Add this to your model card:
```markdown
## Loading Requirements

This NVFP4 model requires patches for compressed-tensors:

\`\`\`bash
pip install git+https://github.com/SILVERTHRONE32/nvfp4-fix.git
nvfp4-fix apply-patch
\`\`\`

**Fast inference with vLLM** (recommended):
\`\`\`python
import os
os.environ['VLLM_USE_V1'] = '0'
from vllm import LLM
llm = LLM(model="username/model", quantization="compressed-tensors")
\`\`\`

**Slow inference with Transformers**:
\`\`\`python
from nvfp4_fix import enable_low_memory_mode
enable_low_memory_mode()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("username/model")
\`\`\`
```

## 🎯 Affected Models

Any model with:
```json
"quantization_config": {
  "format": "nvfp4-pack-quantized"
}
```

Common sources:
- `llmcompressor` quantized models
- NVIDIA TensorRT Model Optimizer outputs
- NVFP4 models on HuggingFace Hub

## 🤝 Contributing

- ⭐ Star if this helped you!
- 🐛 Report issues
- 📢 Share with others
- 💬 Report to [compressed-tensors](https://github.com/vllm-project/compressed-tensors/issues)

## 📄 License

MIT License

## 🙏 Acknowledgments

Fix developed through systematic debugging on 2026-02-16.  
Solves widespread NVFP4 loading issues with compressed-tensors 0.13.0.

---

**Status**: ✅ Working as of Feb 2026 | Use vLLM for best performance
