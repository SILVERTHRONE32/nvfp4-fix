"""Command-line interface for nvfp4-fix."""

import argparse
import sys
from .patches.patcher import apply_patch, is_patched
from .patches.low_memory import enable_low_memory_mode, is_low_memory_enabled

def main():
    parser = argparse.ArgumentParser(
        description="Fix NVFP4 quantized models for compressed-tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply patch to compressed-tensors
  nvfp4-fix apply-patch

  # Enable low-memory mode for 16GB GPUs
  python -c "from nvfp4_fix import enable_low_memory_mode; enable_low_memory_mode()"

  # Check if patch is applied
  nvfp4-fix check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # apply-patch command
    subparsers.add_parser('apply-patch', help='Apply patch to compressed-tensors')
    
    # check command
    subparsers.add_parser('check', help='Check if patch is applied')
    
    args = parser.parse_args()
    
    if args.command == 'apply-patch':
        print("NVFP4 Fix - Patching compressed-tensors\n" + "="*50)
        if apply_patch():
            print("\n✓✓✓ Success! compressed-tensors is now patched.")
            print("\nFor 16GB GPUs, also enable low-memory mode in your code:")
            print("  from nvfp4_fix import enable_low_memory_mode")
            print("  enable_low_memory_mode()")
            return 0
        else:
            print("\n❌ Patch failed.")
            return 1
    
    elif args.command == 'check':
        if is_patched():
            print("✓ Patch is applied!")
            return 0
        else:
            print("❌ Patch NOT applied")
            print("Run: nvfp4-fix apply-patch")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
