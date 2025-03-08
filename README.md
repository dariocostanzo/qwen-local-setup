# qwen-local-setup

# Qwen/QwQ-32B Model Setup Guide

This repository contains setup instructions and code for running the Qwen/QwQ-32B language model locally.

## Prerequisites

- Python 3.10+ installed
- Git installed
- GitHub account
- Sufficient GPU memory (recommended: 24GB+ VRAM for quantized models)

## Installation and Setup

### Using Windows with Git Bash

If you're using Git Bash on Windows (MINGW64), follow these steps:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwen-local-setup.git
cd qwen-local-setup

# Create virtual environment
py -m venv venv

# Activate virtual environment in Git Bash
source venv/Scripts/activate
```

**Note:** You may see permission errors like `Error: [Errno 13] Permission denied` when creating the virtual environment with Git Bash. As long as you see `(venv)` in your prompt, the environment is actually activated and you can proceed despite these errors.

You can verify your environment is working with:
```bash
which python
python -c "import sys; print(sys.prefix)"
```

### Using Windows with PowerShell (Recommended)

For fewer permission issues, PowerShell is recommended:

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwen-local-setup.git
cd qwen-local-setup

# Create virtual environment
py -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy errors, try:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Install Dependencies

After activating your virtual environment:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformer libraries
pip install transformers accelerate

# Install quantization and model loading libraries
pip install bitsandbytes safetensors

# Save dependencies
pip freeze > requirements.txt
```

## Usage

Run the Qwen/QwQ-32B model:

```bash
# Make sure your virtual environment is activated
py run_qwen.py
```

The script will:
1. Check for GPU availability
2. Automatically choose quantization settings based on available memory
3. Download and cache the model
4. Launch an interactive chat interface

Enter your prompts when prompted and type 'exit' to quit.

## Hardware Requirements

- **GPU Memory**: 
  - At least 12GB VRAM with 4-bit quantization
  - 80GB+ VRAM for full precision
- **Disk Space**: At least 70GB for model weights

## Common Issues and Troubleshooting

### Python Not Found Issues

If you see `Python was not found` errors:

1. Try using `py` instead of `python` for all commands
2. Check if Python is in your PATH with `where py` or `where python`
3. Disable Windows Store Python alias:
   - Open Windows Settings
   - Go to Apps → Apps & features → App execution aliases
   - Turn OFF the toggles for "python.exe" and "python3.exe"

### Virtual Environment Permission Errors

If you see `Permission denied` errors when creating or activating the virtual environment:

1. Check if the environment is actually activated (look for `(venv)` in your prompt)
2. Try using PowerShell instead of Git Bash
3. Run Git Bash as administrator
4. Use the `--system-site-packages` flag: `py -m venv venv --system-site-packages`

### GPU/CUDA Issues

If you encounter CUDA-related errors:

1. Verify your GPU is CUDA-compatible: `py -c "import torch; print(torch.cuda.is_available())"`
2. Install the correct version of PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)
3. Update your GPU drivers from the manufacturer's website

### bitsandbytes Errors on Windows

If you see errors related to bitsandbytes:

1. Try the Windows-specific version: `pip install bitsandbytes-windows`
2. If that doesn't work, fall back to CPU-only mode by modifying the script to set `device_map="cpu"` 

## Files in this Repository

- **run_qwen.py**: The main script for running the model
- **requirements.txt**: List of required Python packages
- **.gitignore**: Configured to exclude model cache and virtual environment
- **README.md**: This documentation file

## Contributing

Feel free to submit issues or pull requests if you find bugs or have suggestions for improvements.
