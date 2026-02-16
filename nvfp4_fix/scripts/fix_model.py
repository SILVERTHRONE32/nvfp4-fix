"""Fix NVFP4 quantized models by injecting missing weight_scale buffers."""

from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig
import torch
import glob

def get_model_class(model_path):
    """Determine the appropriate model class."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    if hasattr(config, 'model_type'):
        if 'llava' in config.model_type.lower():
            from transformers import LlavaForConditionalGeneration
            return LlavaForConditionalGeneration
    
    return AutoModelForCausalLM

def fix_nvfp4_model(input_path, output_path, dtype="bfloat16", verbose=True):
    """
    Fix NVFP4 model and save corrected version.
    
    Args:
        input_path: Path to the broken NVFP4 model
        output_path: Path to save the fixed model
        dtype: Target dtype ("bfloat16" or "float16")
        verbose: Print progress messages
    
    Returns:
        bool: True if successful
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if verbose:
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
    
    # Determine dtype
    torch_dtype = getattr(torch, dtype)
    
    # Load model
    if verbose:
        print("\n1. Loading model...")
    ModelClass = get_model_class(input_path)
    model = ModelClass.from_pretrained(
        input_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    if verbose:
        print(f"✓ Loaded with {ModelClass.__name__}")
    
    # Find all safetensors shards
    if verbose:
        print("\n2. Loading weight_scale tensors from safetensors...")
    shard_files = sorted(glob.glob(str(input_path / "model-*.safetensors")))
    
    if not shard_files:
        shard_files = sorted(glob.glob(str(input_path / "*.safetensors")))
    
    if verbose:
        print(f"Found {len(shard_files)} shard(s)")
    
    scale_tensors = {}
    for shard_path in shard_files:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.endswith('.weight_scale'):
                    # Handle different naming conventions
                    model_key = key.replace('language_model.model.layers', 'model.language_model.layers')
                    module_name = model_key.replace('.weight_scale', '')
                    
                    # Convert float8 to target dtype
                    scale = f.get_tensor(key)
                    if scale.dtype == torch.float8_e4m3fn:
                        scale = scale.to(torch_dtype)
                    scale_tensors[module_name] = scale
    
    if verbose:
        print(f"✓ Found {len(scale_tensors)} weight_scale tensors")
    
    # Inject scales
    if verbose:
        print("\n3. Injecting weight_scale buffers...")
    injected = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight_packed') and name in scale_tensors:
            module.register_buffer('weight_scale', scale_tensors[name])
            injected += 1
    
    if verbose:
        print(f"✓ Injected {injected} buffers")
    
    # Test
    if verbose:
        print("\n4. Testing forward pass...")
    input_ids = torch.randint(0, 1000, (1, 10))
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        if verbose:
            print("✓ Forward pass successful!")
    except Exception as e:
        if verbose:
            print(f"❌ Forward pass failed: {e}")
        raise
    
    # Save
    if verbose:
        print(f"\n5. Saving fixed model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    
    # Copy tokenizer and config files
    import shutil
    for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
                 "tokenizer.model", "vocab.json", "merges.txt"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
    
    if verbose:
        print("✓ Done!")
        print(f"\n{'='*60}")
        print(f"Fixed model saved to: {output_path}")
        print(f"You can now use it with transformers directly!")
        print(f"{'='*60}")
    
    return True
