"""Patch compressed-tensors to support NVFP4 weight_scale buffers."""

import sys
import shutil
from pathlib import Path

def find_compressed_tensors_files():
    """Find the compressed_tensors files that need patching."""
    try:
        import compressed_tensors
        package_path = Path(compressed_tensors.__file__).parent
        base_file = package_path / "compressors" / "base.py"
        quantized_base = package_path / "compressors" / "quantized_compressors" / "base.py"
        fp4_file = package_path / "compressors" / "quantized_compressors" / "fp4_quantized.py"
        return base_file, quantized_base, fp4_file
    except ImportError:
        raise ImportError(
            "compressed-tensors not installed. "
            "Install with: pip install compressed-tensors"
        )

def is_patched():
    """Check if compressed-tensors is already patched."""
    try:
        base_file, quantized_base, fp4_file = find_compressed_tensors_files()
        
        # Check base.py for buffer patch
        with open(base_file, 'r') as f:
            base_patched = "# PATCH: Also add named_buffers" in f.read()
        
        # Check quantized base.py for skip_scale patch
        with open(quantized_base, 'r') as f:
            content = f.read()
            skip_patched = "# PATCHED: Don't skip scale for NVFP4" in content
        
        # Check fp4_quantized.py for float8 conversion patch
        with open(fp4_file, 'r') as f:
            content = f.read()
            fp4_patched = "# PATCHED: Convert float8 to avoid promotion errors" in content
        
        return base_patched and skip_patched and fp4_patched
    except Exception:
        return False

def apply_patch(verbose=True):
    """Apply the NVFP4 buffer patch to compressed-tensors."""
    
    base_file, quantized_base, fp4_file = find_compressed_tensors_files()
    
    if verbose:
        print(f"Found compressed-tensors at:")
        print(f"  {base_file}")
        print(f"  {quantized_base}")
        print(f"  {fp4_file}")
    
    if is_patched():
        if verbose:
            print("✓ Already patched!")
        return True
    
    # Patch 1: base.py - Add buffers to decompression
    if verbose:
        print("\nApplying Patch 1: Include buffers in decompression...")
    
    backup_path = str(base_file) + ".backup"
    shutil.copy(base_file, backup_path)
    
    with open(base_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    patched = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if 'for name, parameter in module.named_parameters():' in line and i > 180 and not patched:
            indent = line[:len(line) - len(line.lstrip())]
            i += 1
            new_lines.append(lines[i])
            
            new_lines.append(indent + '# PATCH: Also add named_buffers (e.g., weight_scale)\n')
            new_lines.append(indent + 'for name, buffer in module.named_buffers():\n')
            new_lines.append(indent + '    compressed_data[name] = buffer\n')
            
            for j in range(i+1, len(lines)):
                new_lines.append(lines[j])
            patched = True
            break
    
    if not patched:
        if verbose:
            print("❌ Failed to patch base.py")
        return False
    
    with open(base_file, 'w') as f:
        f.writelines(new_lines)
    
    if verbose:
        print(f"✓ Patched: {base_file}")
    
    # Patch 2: quantized_compressors/base.py - Don't skip weight_scale loading
    if verbose:
        print("\nApplying Patch 2: Force loading of weight_scale...")
    
    backup_path2 = str(quantized_base) + ".backup"
    shutil.copy(quantized_base, backup_path2)
    
    with open(quantized_base, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the _skip_scale method (around line 137-140)
    new_lines = []
    i = 0
    while i < len(lines):
        if 'def _skip_scale(self):' in lines[i]:
            # Replace the entire method
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            new_lines.append(indent + 'def _skip_scale(self):\n')
            new_lines.append(indent + '    # PATCHED: Don\'t skip scale for NVFP4 - we need it!\n')
            new_lines.append(indent + '    return False\n')
            
            # Skip original method lines until next method
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith('def ') and lines[i][0] != ' '):
                if 'def ' in lines[i] and lines[i][:4] == '    ':  # Next method at same level
                    break
                i += 1
            continue
        else:
            new_lines.append(lines[i])
            i += 1
    
    with open(quantized_base, 'w') as f:
        f.writelines(new_lines)
    
    if verbose:
        print(f"✓ Patched: {quantized_base}")
    
    # Patch 3: fp4_quantized.py - Convert Float8 to BFloat16
    if verbose:
        print("\nApplying Patch 3: Handle Float8 dtype conversion...")
    
    backup_path3 = str(fp4_file) + ".backup"
    shutil.copy(fp4_file, backup_path3)
    
    with open(fp4_file, 'r') as f:
        content = f.read()
    
    # Find and replace the scale loading line
    old_code = '        scale = compressed_data["weight_scale"]'
    
    new_code = '''        scale = compressed_data["weight_scale"]
        # PATCHED: Convert float8 to avoid promotion errors
        import torch
        if scale.dtype == torch.float8_e4m3fn:
            scale = scale.to(torch.bfloat16)'''
    
    content = content.replace(old_code, new_code)
    
    with open(fp4_file, 'w') as f:
        f.write(content)
    
    if verbose:
        print(f"✓ Patched: {fp4_file}")
    
    if verbose:
        print(f"\n📦 Backups created:")
        print(f"  {backup_path}")
        print(f"  {backup_path2}")
        print(f"  {backup_path3}")
    
    return True

if __name__ == "__main__":
    print("NVFP4 Fix - Patching compressed-tensors\n" + "="*50)
    
    if apply_patch():
        print("\n✓✓✓ Success! compressed-tensors is now patched.")
        print("\nYou can now load NVFP4 models directly!")
    else:
        print("\n❌ Patch failed. Please report this issue.")
        sys.exit(1)
