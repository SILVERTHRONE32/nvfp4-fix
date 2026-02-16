# NVFP4 Fix for compressed-tensors

Fix `KeyError: 'weight_scale'` when loading NVFP4 quantized models with HuggingFace Transformers.

## 🚀 Quick Start
```bash
# Install
pip install git+https://github.com/yourusername/nvfp4-fix.git

# Apply patch to compressed-tensors (one-time setup)
nvfp4-fix apply-patch

# Fix your model
nvfp4-fix fix-model /path/to/broken-model /path/to/fixed-model

# Use the fixed model
python -c "from transformers import AutoModelForCausalLM; \
           model = AutoModelForCausalLM.from_pretrained('/path/to/fixed-model')"
```

## 🐛 What This Fixes

- ✅ `KeyError: 'weight_scale'` during model loading
- ✅ `RuntimeError: Promotion for Float8 Types is not supported`  
- ✅ NVFP4 models from llmcompressor, NVIDIA TensorRT Model Optimizer
- ✅ Works with LLaVA, Mistral, Llama, Gemma, and other architectures

## 📖 Problem Background

NVFP4 (NVIDIA FP4) quantized models store per-group quantization scales as **buffers** in safetensors files. However, compressed-tensors v0.13.0's `decompress_module()` only extracts **parameters**, causing the `weight_scale` buffers to be ignored during decompression.

Additionally, these scales are stored in `float8_e4m3fn` format which conflicts with model dtypes like `bfloat16`.

## 🔧 How It Works

### 1. Patches compressed-tensors
Modifies `compressed_tensors/compressors/base.py` to include buffers:
```python
# Before (broken)
for name, parameter in module.named_parameters():
    compressed_data[name] = parameter

# After (fixed)
for name, parameter in module.named_parameters():
    compressed_data[name] = parameter
for name, buffer in module.named_buffers():
    compressed_data[name] = buffer  # ← includes weight_scale!
```

### 2. Injects Missing Scales
Reads `weight_scale` tensors from safetensors and registers them as buffers in the model.

### 3. Converts Float8 → BFloat16
Converts `float8_e4m3fn` scales to `bfloat16` to avoid dtype promotion errors.

## 📦 Installation
```bash
pip install git+https://github.com/yourusername/nvfp4-fix.git
```

Or for development:
```bash
git clone https://github.com/yourusername/nvfp4-fix.git
cd nvfp4-fix
pip install -e .
```

## 🎯 Usage

### CLI
```bash
# Check if patch is applied
nvfp4-fix check

# Apply patch (one-time)
nvfp4-fix apply-patch

# Fix a model
nvfp4-fix fix-model input_model/ output_model/ --dtype bfloat16
```

### Python API
```python
from nvfp4_fix import apply_patch, fix_nvfp4_model, is_patched

# Check and apply patch
if not is_patched():
    apply_patch()

# Fix a model
fix_nvfp4_model(
    input_path="/path/to/broken-model",
    output_path="/path/to/fixed-model",
    dtype="bfloat16"
)
```

## 📤 Uploading Fixed Models to HuggingFace

**Important:** Fixed models only work out-of-the-box for users who have also applied this patch.

Add this to your model card:
```markdown
## ⚠️ Loading Requirements

This model requires the NVFP4 fix for compressed-tensors:

\`\`\`bash
pip install git+https://github.com/yourusername/nvfp4-fix.git
nvfp4-fix apply-patch
\`\`\`

Then load normally:
\`\`\`python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
\`\`\`

See [nvfp4-fix](https://github.com/yourusername/nvfp4-fix) for details.
```

## 🎯 Affected Models

Any model with this in `config.json`:
```json
"quantization_config": {
  "format": "nvfp4-pack-quantized"
}
```

Common sources:
- Models quantized with `llmcompressor`
- Models from NVIDIA TensorRT Model Optimizer
- NVFP4 versions on HuggingFace Hub

## 🧪 Testing

Verify the fix works:
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/fixed-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test inference
inputs = torch.randint(0, 1000, (1, 10)).to(model.device)
outputs = model(inputs)
print("✓ Model works!")
```

## 🤝 Contributing

Issues and PRs welcome! This is a community fix for a widespread problem.

Please:
- ⭐ Star this repo if it helped you
- 🐛 Report bugs in [Issues](https://github.com/yourusername/nvfp4-fix/issues)
- 📢 Share with others hitting this error
- 💬 Report to [compressed-tensors](https://github.com/vllm-project/compressed-tensors/issues)

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Fix developed through collaborative debugging session on 2026-02-16.  
Affects widespread NVFP4 deployments with compressed-tensors 0.13.0.

---

**Status:** ✅ Working as of Feb 2026 | Temporary workaround until official fix
