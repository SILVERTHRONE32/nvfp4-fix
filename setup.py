from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nvfp4-fix",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fix NVFP4 quantized models for compressed-tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nvfp4-fix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "compressed-tensors>=0.13.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "torch>=2.0.0",
    ],
    entry_points={
        'console_scripts': [
            'nvfp4-fix=nvfp4_fix.cli:main',
        ],
    },
)
