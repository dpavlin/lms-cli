# LM Studio CLI Utility

A comprehensive Python-based CLI utility for interacting with [LM Studio](https://lmstudio.ai/). This tool provides an easy-to-use interface for managing models, performing chat/completions, and generating configurations for external tools like OpenCode.

## Features

- **Model Management**: List, search, load, and unload models with specific configurations.
- **Dynamic Configuration**: Change context length and GPU offload settings on the fly.
- **Interactions**: Perform single-turn chat, text completions, and generate embeddings.
- **Status Monitoring**: Check server availability and see currently loaded models with their allocated context sizes.
- **Integrations**: Automatically generate `opencode.json` for [OpenCode](https://opencode.ai/) with specialized agents for planning (thinking models) and execution (coder models).
- **Utility Tools**: Quick access to system prompt templates for common tasks (Coder, Logic, Creative, etc.).

## Installation

1. Ensure you have Python 3 installed.
2. Clone this repository or copy `lms_cli.py` to your local machine.
3. Make the script executable:
   ```bash
   chmod +x lms_cli.py
   ```

## Usage

### Configuration
By default, the CLI looks for LM Studio at `http://localhost:1234`.
```bash
# Update the base URL
./lms_cli.py config --url http://192.168.1.10:1234

# Show current config
./lms_cli.py config --show
```

### Model Management
```bash
# List available models
./lms_cli.py list

# Load a model with 32k context and full GPU offload
./lms_cli.py load <model_id> --context 32768 --gpu 1.0

# Unload a specific instance
./lms_cli.py unload <instance_id>
```

### Interactions
```bash
# Quick chat with a system prompt
./lms_cli.py chat <model_id> "Refactor this code..." --system "You are a senior Rust engineer."

# Get useful templates
./lms_cli.py templates
```

### Integrations
```bash
# Generate OpenCode config
./lms_cli.py opencode > opencode.json
```

## Requirements
- Python 3.x
- LM Studio (with Local Server enabled)
