# LM Studio CLI Utility

A professional Python-based CLI utility for interacting with [LM Studio](https://lmstudio.ai/). This tool provides a powerful interface for model management, streaming interactions, performance benchmarking, and hardware-aware model discovery.

## Features

- **Advanced Model Management**:
    - List, search, load, and unload models.
    - **`unload --all`**: Instantly clear all models from VRAM.
    - **`download`**: Auto-converts Hugging Face repo IDs to full URLs.
    - **`download-status`**: Track background download progress.
- **Hardware-Aware Search**:
    - **VRAM Estimation**: Automatically estimates if a model will fit on your GPU based on your configured VRAM capacity.
    - **Hugging Face Discovery**: Search the HF Hub for new GGUF models directly from your terminal.
- **High-Performance Interactions**:
    - **`chat --stream`**: Real-time token streaming for chat.
    - **`repl`**: Interactive, stateful, streaming conversation mode.
    - **`bench`**: Measure **Time to First Token (TTFT)** and **Tokens Per Second (TPS)**.
- **Dynamic Configuration**:
    - Set `base_url`, `timeout`, and `vram_gb` (for fit estimations).
    - Configure `context_length` and `gpu_layers` per model load.
- **Integrations & Utilities**:
    - **OpenCode**: Automatically generate `opencode.json` with auto-detected Planner (thinking) and Coder models.
    - **Presets**: List local LM Studio configuration presets.
    - **Templates**: Quick access to optimized system prompts.

## Installation

1. Ensure you have Python 3 installed.
2. Clone this repository or copy `lms_cli.py` to your local machine.
3. Make the script executable:
   ```bash
   chmod +x lms_cli.py
   ```

## Usage

### Configuration
Set up your environment and hardware limits.
```bash
# Set your GPU's VRAM capacity (used for fit estimation in 'search')
./lms_cli.py config --vram 12.0

# Show current config
./lms_cli.py config --show
```

### Model Discovery & Download
Find models that fit your hardware and download them.
```bash
# Search local and remote models with VRAM fit indicator
./lms_cli.py search "llama-3.2"

# Download a model using its repo identifier
./lms_cli.py download bartowski/Llama-3.2-1B-Instruct-GGUF

# Check progress
./lms_cli.py download-status
```

### Model Management
Control what's in your GPU memory.
```bash
# Load with specific context and full GPU offload
./lms_cli.py load <model_id> --context 32768 --gpu 1.0

# Watch loaded models in real-time (dashboard mode)
./lms_cli.py status --watch

# Clear all models from memory
./lms_cli.py unload --all
```

### Performance & Interaction
Chat or test your system's speed.
```bash
# Benchmark model speed (TTFT/TPS)
./lms_cli.py bench <model_id>

# Start an interactive, streaming session
./lms_cli.py repl <model_id> --system "You are a helpful assistant."

# Quick streaming chat
./lms_cli.py chat <model_id> "Write a hello world in Python" --stream
```

### Integrations
Generate configs for tools like OpenCode.
```bash
# Auto-detect coder and thinking models and generate config
./lms_cli.py opencode > opencode.json
```

## Requirements
- Python 3.x
- LM Studio (with Local Server enabled)