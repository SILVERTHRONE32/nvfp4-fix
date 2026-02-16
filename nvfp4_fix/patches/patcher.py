"""Patch compressed-tensors to support NVFP4 weight_scale buffers."""

import sys
import shutil
from pathlib import Path

def find_compressed_tensors_base():
    """Find the compressed_tensors base.py file."""
    try:
        import compressed_tensors
        package_path = Path(compressed_tensors.__file__).parent
        return package_path / "compressors" / "base.py"
    except ImportError:
        raise ImportError(
            "compressed-tensors not installed. "
            "Install with: pip install compressed-tensors"
        )

def is_patched():
    """Check if compressed-tensors is already patched."""
    try:
        base_file = find_compressed_tensors_base()
        with open(base_file, 'r') as f:
            content = f.read()
        return "# PATCH: Also add named_buffers" in content
    except Exception:
        return False

def apply_patch(verbose=True):
    """Apply the NVFP4 buffer patch to compressed-tensors."""
    
    base_file = find_compressed_tensors_base()
    
    if verbose:
        print(f"Found compressed-tensors at: {base_file}")
    
    if is_patched():
        if verbose:
            print("✓ Already patched!")
        return True
    
    # Backup
    backup_path = str(base_file) + ".backup"
    shutil.copy(base_file, backup_path)
    if verbose:
        print(f"📦 Backup created: {backup_path}")
    
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
            
            # Add the patch
            new_lines.append(indent + '# PATCH: Also add named_buffers (e.g., weight_scale)\n')
            new_lines.append(indent + 'for name, buffer in module.named_buffers():\n')
            new_lines.append(indent + '    compressed_data[name] = buffer\n')
            
            for j in range(i+1, len(lines)):
                new_lines.append(lines[j])
            patched = True
            break
    
    if not patched:
        if verbose:
            print("❌ Failed to find patch location. File structure may have changed.")
        return False
    
    with open(base_file, 'w') as f:
        f.writelines(new_lines)
    
    if verbose:
        print(f"✓ Patched: {base_file}")
    return True

if __name__ == "__main__":
    print("NVFP4 Fix - Patching compressed-tensors\n" + "="*50)
    
    if apply_patch():
        print("\n✓✓✓ Success! compressed-tensors is now patched.")
        print("\nNext: Use 'nvfp4-fix' command to fix your models")
    else:
        print("\n❌ Patch failed. Please report this issue.")
        sys.exit(1)
