# LM Studio CLI Utility

A high-performance Python CLI for [LM Studio](https://lmstudio.ai/). Designed for developers who need fast model management, real-time monitoring, and hardware-aware model discovery.

## ✨ Key Smart Features

- **Interactive Switcher**: Use `./lms_cli.py switch` to browse your models, see their VRAM footprint, and load them interactively with automatic memory management.
- **Real-time Dashboard**: Use `./lms_cli.py top` for a `top`-like view of your LM Studio server, including loaded models and background download progress bars.
- **Context-Aware Commands**: Most commands (like `chat`, `repl`, `bench`) **automatically use the currently active model** if you don't specify one.
- **Hardware Estimation**: The `search` and `list` commands estimate VRAM usage and show a 🟢/🟡/🔴 indicator based on your GPU's capacity.
- **Intelligent Discovery**: Search the Hugging Face Hub for GGUF models directly from your terminal with tool-support (🛠️) detection.

## 🛠️ Installation

1. Ensure Python 3 is installed.
2. Make the script executable:
   ```bash
   chmod +x lms_cli.py
   ```

## 🚀 Quick Start

### 1. Configuration
Set your hardware limits to enable "Smart Fit" estimation.
```bash
# Tell the CLI your GPU has 12GB VRAM
./lms_cli.py config --vram 12.0 --context 32768
```

### 2. Model Management
```bash
# Interactive model selection and loading
./lms_cli.py switch

# Real-time dashboard
./lms_cli.py top

# Clear all models from GPU immediately
./lms_cli.py unload --all
```

### 3. Interaction (Active Model)
If a model is loaded, you don't need to provide its ID.
```bash
# Start a streaming, stateful conversation
./lms_cli.py repl

# Benchmark your GPU's generation speed (TTFT/TPS)
./lms_cli.py bench

# Quick chat with a specific persona
./lms_cli.py chat "Explain quantum gravity" --system "You are a friendly cat." --stream
```

### 4. Discovery & Download
```bash
# Find tool-capable models that fit your GPU
./lms_cli.py search "qwen3 coder"

# Download using repo ID (auto-converts to HF URL)
./lms_cli.py download bartowski/Llama-3.2-1B-Instruct-GGUF
```

### 5. Integrations
```bash
# Generate OpenCode config with auto-detected Planner/Coder models
./lms_cli.py opencode --context 32768 > opencode.json
```

## 📋 Command Reference

| Command | Description |
| :--- | :--- |
| `config` | Manage CLI settings like base URL, VRAM, and default context. |
| `status` | Show server summary and currently loaded models. |
| `list` | Detailed table of local models (alias: `models`). |
| `switch` | Interactive list and load interface with VRAM management. |
| `top` | Real-time server dashboard and background download tracker. |
| `check` | Quick connectivity check to the LM Studio server. |
| `info` | Get detailed technical metadata for a specific model. |
| `load` | Load a model into memory with optional settings. |
| `unload` | Clear specific models or all models from VRAM. |
| `search` | Discover models on Hugging Face with "Smart Fit" indicators. |
| `download` | Download models from Hugging Face (auto-URL conversion). |
| `download-status` | Track progress of active background downloads. |
| `presets` | List local LM Studio configuration presets. |
| `chat` | Send a single message (supports streaming and system prompts). |
| `repl` | Start an interactive, stateful, streaming conversation session. |
| `bench` | Measure model performance (TTFT and Tokens Per Second). |
| `complete` | Perform classic non-chat text completion. |
| `embeddings` | Generate vector embeddings for text input. |
| `opencode` | Generate highly compatible `opencode.json` config files. |
| `templates` | Show a collection of optimized system prompt templates. |
| `raw` | Send arbitrary HTTP requests to the LM Studio API. |

## Requirements
- Python 3.x
- LM Studio (Local Server enabled on port 1234)
