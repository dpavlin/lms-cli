# LM Studio CLI (vibe coded with gemini)

Just a quick and dirty Python script to manage [LM Studio](https://lmstudio.ai/) from your terminal without having to click around the GUI all the time. 

It was built entirely with vibes and Gemini (Google's AI), so it's probably got some quirks, but it works surprisingly well for moving models around and checking what's going on.

## ✨ The Cool Stuff

- **`switch`**: Interactively browse your models, see if they'll explode your VRAM, and load them. It unloads everything else first so you don't OOM.
- **`top`**: A dashboard that actually shows you what's loaded and how your background downloads are doing.
- **Lazy Commands**: If you have a model loaded, just type `./lms_cli.py chat "hi"` and it'll figure out which model to use.
- **Smart Search**: Search Hugging Face for GGUF models and see a 🟢/🟡/🔴 indicator based on your GPU size.
- **OpenCode helper**: Generates `opencode.json` with the right settings so you can use thinking models for planning and coder models for building.

## 🛠️ How to use it

1. Make it executable: `chmod +x lms_cli.py`
2. Tell it about your GPU: `./lms_cli.py config --vram 12.0 --context 32768`
3. Just run it: `./lms_cli.py switch`

## 🚀 Quick Examples

```bash
# Benchmark your speed
./lms_cli.py bench

# Chat with streaming
./lms_cli.py chat "How do I fix my life?" --stream

# Search for a new model to try
./lms_cli.py search "deepseek coder"

# Panic button (clears VRAM)
./lms_cli.py unload --all
```

## 📋 Every Command

| Command | What it does |
| :--- | :--- |
| `config` | Tell the script about your URL, VRAM, and preferred context size. |
| `status` | Who's loaded right now? |
| `list` | Show everything you've downloaded so far (alias: `models`). |
| `switch` | The "best" way to pick and load a model. |
| `top` | Live dashboard for vibes and downloads. |
| `check` | Is the server even on? |
| `info` | Nerd stats about a model (arch, quant, etc.). |
| `load` | Force load a model with specific settings. |
| `unload` | Kick a model (or all of them) out of VRAM. |
| `search` | Find new models on Hugging Face that actually fit your card. |
| `download` | Grab a model from HF (you can just paste the `user/repo`). |
| `download-status` | How much longer for that 20GB file? |
| `presets` | Show those LM Studio config presets. |
| `chat` | Quick one-off message. |
| `repl` | Full-on conversation mode. |
| `bench` | Speed test (TTFT/TPS) with GPU detection. |
| `complete` | Classic text completion. |
| `embeddings` | Turn text into numbers. |
| `opencode` | Fix those context issues for OpenCode. |
| `templates` | System prompt cheat sheet. |
| `raw` | Send whatever JSON you want to the API. |

## Requirements
- Python 3
- LM Studio running with the Local Server turned on (usually port 1234)

---
*Built with ❤️ and a lot of back-and-forth with Gemini.*