"""Command-line interface for nvfp4-fix."""

import argparse
import sys
from .patches.patcher import apply_patch, is_patched
from .scripts.fix_model import fix_nvfp4_model

def main():
    parser = argparse.ArgumentParser(
        description="Fix NVFP4 quantized models for compressed-tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply patch to compressed-tensors
  nvfp4-fix apply-patch

  # Fix a model
  nvfp4-fix fix-model /path/to/broken-model /path/to/fixed-model

  # Check if patch is applied
  nvfp4-fix check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # apply-patch command
    subparsers.add_parser('apply-patch', help='Apply patch to compressed-tensors')
    
    # check command
    subparsers.add_parser('check', help='Check if patch is applied')
    
    # fix-model command
    fix_parser = subparsers.add_parser('fix-model', help='Fix an NVFP4 model')
    fix_parser.add_argument('input_path', help='Path to broken NVFP4 model')
    fix_parser.add_argument('output_path', help='Path to save fixed model')
    fix_parser.add_argument('--dtype', default='bfloat16', 
                           choices=['bfloat16', 'float16'],
                           help='Target dtype (default: bfloat16)')
    
    args = parser.parse_args()
    
    if args.command == 'apply-patch':
        print("NVFP4 Fix - Patching compressed-tensors\n" + "="*50)
        if apply_patch():
            print("\n✓✓✓ Success! compressed-tensors is now patched.")
            print("\nNext: Use 'nvfp4-fix fix-model' to fix your models")
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
    
    elif args.command == 'fix-model':
        try:
            if not is_patched():
                print("⚠️  Warning: Patch not applied to compressed-tensors!")
                print("   Run 'nvfp4-fix apply-patch' first")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return 1
            
            fix_nvfp4_model(args.input_path, args.output_path, args.dtype)
            return 0
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
