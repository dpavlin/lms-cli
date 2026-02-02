# LM Studio CLI (vibe coded with gemini)

Just a quick and dirty Python script to manage [LM Studio](https://lmstudio.ai/) from your terminal without having to click around the GUI all the time. 

It was built entirely with vibes and Gemini (Google's AI), so it's probably got some quirks, but it works surprisingly well for moving models around and checking what's going on.

## ✨ The Cool Stuff

- **`switch`**: Interactively browse your models, filter by name, and load them. It automatically unloads everything else first to save VRAM and sends a "ping" to verify the model is actually awake.
- **`top`**: A real-time dashboard that shows you what's loaded (with VRAM/Context stats) and how your background downloads are doing.
- **`tool-test`**: A rigorous stress-test to see if a model actually supports **Parallel Tool Calling** (essential for agents like OpenCode).
- **Lazy Commands**: If you have a model loaded, just type `./lms_cli.py chat "hi"` and it'll figure out which model to use.
- **Smart Search**: Search Hugging Face for GGUF models and see a 🟢/🟡/🔴 indicator based on your GPU size and 🛠️ icons for tool support.

## 🛠️ How to use it

1. Make it executable: `chmod +x lms_cli.py`
2. Tell it about your GPU: `./lms_cli.py config --vram 16.0 --context 32768`
3. Just run it: `./lms_cli.py switch`

## 🚀 Quick Examples

```bash
# Benchmark your speed & detect GPU acceleration
./lms_cli.py bench

# Test if your current model can handle complex agent tasks
./lms_cli.py tool-test

# Search for a specific model class
./lms_cli.py search "qwen3 coder"

# Panic button (clears VRAM)
./lms_cli.py unload --all
```

## 📋 Every Command

| Command | What it does |
| :--- | :--- |
| `config` | Set your URL, VRAM capacity, and default context size (32k by default). |
| `status` | The big dashboard. Who's loaded? What are their specs (Arch, Quant, Context)? |
| `list` | Show everything you've downloaded so far (alias: `models`). |
| `switch` | The "best" way to pick and load a model. Supports optional filtering. |
| `top` | Live auto-refreshing dashboard for vibes and downloads. |
| `check` | Is the server even on? |
| `tool-test` | Can this model handle file edits and shell commands? |
| `load` | Force load a model with specific context/gpu settings. |
| `unload` | Kick a model (or all of them) out of VRAM. |
| `search` | Find new models on Hugging Face that actually fit your card. |
| `download` | Grab a model from HF. Auto-switches to the model when done! |
| `download-status` | How much longer for that 20GB file? |
| `presets` | Show those LM Studio config presets. |
| `chat` | Quick one-off message (auto-uses active model). |
| `repl` | Full-on streaming conversation mode. |
| `bench` | Speed test (TTFT/TPS) with heuristic GPU detection. |
| `complete` | Classic text completion. |
| `embeddings` | Turn text into numbers. |
| `opencode` | Generate `opencode.json` with 32k context and tool flags. |
| `templates` | System prompt cheat sheet. |
| `raw` | Send whatever JSON you want to the API. |

## Requirements
- Python 3
- LM Studio running with the Local Server turned on (usually port 1234)

---
*Built with ❤️ and a lot of back-and-forth with Gemini.*
