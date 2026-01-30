#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
import re
from typing import Optional, Dict, Any, List, Iterator

# Constants
CONFIG_DIR = os.path.expanduser("~/.config/lms-cli")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_CONFIG = {
    "base_url": "http://localhost:1234",
    "timeout": 30,
    "vram_gb": 12.0
}
PRESETS_DIR = os.path.expanduser("~/.lmstudio/config-presets")

class ConfigManager:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(CONFIG_FILE):
            return DEFAULT_CONFIG.copy()
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception as e:
            print(f"Warning: Could not load config: {e}", file=sys.stderr)
            return DEFAULT_CONFIG.copy()

    def save_config(self, key: str, value: Any):
        self.config[key] = value
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration updated: {key} = {value}")

    def get(self, key: str) -> Any:
        return self.config.get(key)

class LMStudioClient:
    def __init__(self, base_url: str, timeout: int, vram_gb: float = 12.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.vram_gb = vram_gb

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, stream: bool = False) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, method=method, headers=headers)
        if data:
            req.data = json.dumps(data).encode('utf-8')
        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
            if stream:
                return response
            return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            print(f"HTTP Error {e.code}: {body}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"Error connecting to {url}: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response from {url}", file=sys.stderr)
            sys.exit(1)

    def check(self):
        try:
            self._request("GET", "/v1/models")
            print(f"✔ LM Studio is reachable at {self.base_url}")
            return True
        except:
            print(f"✘ LM Studio is NOT reachable at {self.base_url}")
            return False

    def status(self, watch: bool = False):
        try:
            while True:
                if watch: os.system('clear')
                print(f"--- LM Studio Status ---")
                print(f"Base URL: {self.base_url}")
                try:
                    data = self._request("GET", "/api/v0/models")
                    models = data.get('data', [])
                    loaded = [m for m in models if m.get('state') == 'loaded']
                    print(f"Status:   Online")
                    print(f"Models:   {len(models)} available")
                    if loaded:
                        print(f"\nCurrently Loaded Models:")
                        for m in loaded:
                            ctx = m.get('loaded_context_length', 'unknown')
                            print(f" - {m.get('id')} ({m.get('type', 'llm')}, context: {ctx})")
                    else:
                        print("\nNo models currently loaded.")
                except:
                    print(f"Status:   Offline")
                print("\nNote: System resources (CPU/GPU/RAM) are not exposed via API.")
                if not watch: break
                time.sleep(2)
        except KeyboardInterrupt: pass

    def list_models(self):
        print(f"Available Models at {self.base_url}:")
        try:
            data = self._request("GET", "/v1/models")
            for model in data.get('data', []):
                print(f" - {model.get('id')}")
        except Exception as e:
            print(f"Error listing models: {e}")

    def info(self, model_id: Optional[str] = None):
        try:
            data = self._request("GET", "/api/v0/models")
            models = data.get('data', [])
            
            if model_id:
                model_list = [m for m in models if m['id'] == model_id]
                if not model_list:
                    print(f"Model '{model_id}' not found.")
                    return
            else:
                model_list = [m for m in models if m.get('state') == 'loaded']
                if not model_list:
                    print("No models are currently loaded.")
                    return
                print(f"--- Info for all {len(model_list)} loaded model(s) ---")

            for model in model_list:
                if len(model_list) > 1: print(f"\n--- {model['id']} ---")
                print(f"Model: {model['id']}")
                print(f"  Type:         {model.get('type', 'N/A')}")
                print(f"  Architecture: {model.get('arch', 'N/A')}")
                print(f"  Quantization: {model.get('quantization', 'N/A')}")
                print(f"  Max Context:  {model.get('max_context_length', 'N/A')}")
                print(f"  Publisher:    {model.get('publisher', 'N/A')}")
                print(f"  State:        {model.get('state', 'N/A')}")
                
                caps = model.get('capabilities', [])
                if caps:
                    print(f"  Capabilities: {', '.join(caps)}")
                else:
                    print(f"  Capabilities: None reported by LM Studio")
        except Exception as e:
            print(f"Error: {e}")

    def load(self, model_id: str, context: Optional[int] = None, gpu: Optional[float] = None):
        print(f"Loading {model_id}...", file=sys.stderr)
        payload = {"model": model_id}
        if context: payload["context_length"] = context
        if gpu is not None:
            payload["gpu_layers"] = int(gpu) if gpu > 1 else -1
        try:
            resp = self._request("POST", "/api/v1/models/load", payload)
            print(f"Loaded instance: {resp.get('instance_id')}")
        except Exception as e:
            print(f"Load failed: {e}")
            print("Falling back to chat completion load...")
            fb = {"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
            self._request("POST", "/v1/chat/completions", fb)
            print("Load triggered (fallback).")

    def unload(self, identifier: str = None, all_models: bool = False):
        if all_models:
            print("Unloading all models...")
            try:
                data = self._request("GET", "/api/v0/models")
                for m in data.get('data', []):
                    if m.get('state') == 'loaded':
                        print(f"Unloading {m['id']}...")
                        try:
                            self._request("POST", "/api/v1/models/unload", {"instance_id": m['id']})
                        except: pass
                print("Done.")
            except Exception as e:
                print(f"Failed to list models for unloading: {e}")
        elif identifier:
            print(f"Unloading {identifier}...")
            try:
                self._request("POST", "/api/v1/models/unload", {"instance_id": identifier})
                print("Unload successful.")
            except Exception:
                try:
                    self._request("POST", "/api/v1/models/unload", {"model": identifier})
                    print("Unload successful (fallback).")
                except:
                    print("Unload failed. Ensure you use the correct ID.")

    def download(self, model_id: str):
        if not model_id.startswith("http") and "/" in model_id:
            print(f"Converting '{model_id}' to Hugging Face URL...")
            model_id = f"https://huggingface.co/{model_id}"
        print(f"Requesting download for: {model_id}")
        try:
            resp = self._request("POST", "/api/v1/models/download", {"model": model_id})
            print(f"Download started: {json.dumps(resp, indent=2)}")
            if "job_id" in resp:
                print(f"\nTrack progress with: ./lms_cli.py download-status {resp['job_id']}")
        except Exception as e: print(f"Download failed: {e}")

    def download_status(self, job_id: str = None):
        endpoint = "/api/v1/models/download/status"
        if job_id: endpoint += f"/{job_id}"
        try:
            data = self._request("GET", endpoint)
            print("Download Status:")
            print(json.dumps(data, indent=4))
        except:
            print("No active download found or endpoint unavailable.")

    def search(self, query: str):
        vram_limit = self.vram_gb
        try:
            data = self._request("GET", "/v1/models")
            local = [m['id'] for m in data.get('data', []) if query.lower() in m['id'].lower()]
            if local:
                print(f"--- Local Models matching '{query}' ---")
                for m in local: print(f" - {m}")
                print()
        except: pass

        print(f"--- Searching Hugging Face for '{query}' (GGUF) ---")
        print(f" (Estimating fit for {vram_limit}GB VRAM)")
        encoded_query = urllib.parse.quote(query)
        hf_url = f"https://huggingface.co/api/models?search={encoded_query}&filter=gguf&sort=downloads&direction=-1&limit=10"
        try:
            req = urllib.request.Request(hf_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                models = json.loads(response.read().decode('utf-8'))
                if not models:
                    print("No models found on Hugging Face.")
                    return
                for m in models:
                    m_id = m.get('modelId')
                    downloads = m.get('downloads', 0)
                    match = re.search(r'(\d+)b', m_id.lower())
                    vram_est = "???"
                    fit_icon = "⚪"
                    if match:
                        params = int(match.group(1))
                        est_gb = (params * 0.6) + 1.5
                        vram_est = f"~{est_gb:.1f}GB"
                        if est_gb <= vram_limit: fit_icon = "🟢"
                        elif est_gb <= vram_limit * 1.2: fit_icon = "🟡"
                        else: fit_icon = "🔴"
                    print(f" {fit_icon} {m_id:<50} | {vram_est:>7} | ↓ {downloads}")
                print("\nLegend: 🟢 Fits  🟡 Tight  🔴 Likely OOM  ⚪ Unknown")
        except Exception as e: print(f"HF search failed: {e}")

    def chat(self, model_id: str, message: str, system: Optional[str] = None, 
             stream: bool = False, temp: float = 0.7, max_tokens: int = -1, top_p: float = 1.0):
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": message})
        payload = {
            "model": model_id, 
            "messages": msgs, 
            "temperature": temp, 
            "top_p": top_p,
            "stream": stream
        }
        if max_tokens > 0: payload["max_tokens"] = max_tokens

        try:
            if not stream:
                resp = self._request("POST", "/v1/chat/completions", payload)
                print(f"\nResponse:\n{resp.get('choices')[0].get('message').get('content')}\n")
            else:
                response = self._request("POST", "/v1/chat/completions", payload, stream=True)
                print("\nResponse: ", end="", flush=True)
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        if line == "data: [DONE]": break
                        try:
                            chunk = json.loads(line[6:])
                            content = chunk['choices'][0]['delta'].get('content', '')
                            print(content, end="", flush=True)
                        except: pass
                print("\n")
        except Exception as e:
            print(f"Chat failed: {e}")

    def repl(self, model_id: str, system: Optional[str] = None):
        print(f"Entering REPL with {model_id}. Type 'exit' or 'quit' to end.")
        history = []
        if system: history.append({"role": "system", "content": system})
        try:
            while True:
                user_input = input("\n>>> ")
                if not user_input.strip(): continue
                if user_input.lower() in ['exit', 'quit']: break
                history.append({"role": "user", "content": user_input})
                
                payload = {"model": model_id, "messages": history, "stream": True}
                try:
                    response = self._request("POST", "/v1/chat/completions", payload, stream=True)
                    full_response = ""
                    print("Assistant: ", end="", flush=True)
                    for line in response:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            if line == "data: [DONE]": break
                            try:
                                chunk = json.loads(line[6:])
                                content = chunk['choices'][0]['delta'].get('content', '')
                                print(content, end="", flush=True)
                                full_response += content
                            except: pass
                    print()
                    history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    print(f"REPL Error: {e}")
        except (KeyboardInterrupt, EOFError): print("\nExiting REPL.")

    def bench(self, model_id: str):
        print(f"Benchmarking {model_id}...")
        prompt = "Explain quantum entanglement in simple terms."
        payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "stream": True}
        start_time = time.time()
        ttft = 0
        token_count = 0
        try:
            response = self._request("POST", "/v1/chat/completions", payload, stream=True)
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    if line == "data: [DONE]": break
                    if ttft == 0: ttft = time.time() - start_time
                    token_count += 1
            total_time = time.time() - start_time
            tps = token_count / (total_time - ttft) if (total_time - ttft) > 0 else 0
            print(f"\nResults: TTFT: {ttft:.4f}s | TPS: {tps:.2f} | Tokens: {token_count} | Total: {total_time:.4f}s")
        except Exception as e:
            print(f"Benchmark failed: {e}")

    def complete(self, model_id: str, prompt: str):
        print(f"Completing with {model_id}...")
        try:
            resp = self._request("POST", "/v1/completions", {"model": model_id, "prompt": prompt})
            print(f"\nResult:\n{resp.get('choices')[0].get('text')}\n")
        except Exception as e:
            print(f"Completion failed: {e}")

    def embeddings(self, model_id: str, input_text: str):
        print(f"Generating embeddings with {model_id}...")
        try:
            resp = self._request("POST", "/v1/embeddings", {"model": model_id, "input": input_text})
            vec = resp.get('data')[0].get('embedding')
            print(f"Vector length: {len(vec)}\nFirst 5: {vec[:5]}")
        except Exception as e:
            print(f"Embeddings failed: {e}")

    def presets(self):
        if not os.path.exists(PRESETS_DIR):
            print(f"Presets directory not found: {PRESETS_DIR}")
            return
        files = [f for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
        if not files:
            print("No presets found.")
        else:
            print("Local Presets:")
            for f in files: print(f" - {f[:-5]}")

    def opencode(self, coder_id: Optional[str] = None, thinking_id: Optional[str] = None, context: int = 32768):
        print("Generating OpenCode configuration...", file=sys.stderr)
        try:
            data = self._request("GET", "/api/v0/models")
            models = data.get('data', [])
            cfg = {
                "$schema": "https://opencode.ai/config.json", 
                "provider": {
                    "lmstudio": {
                        "npm": "@ai-sdk/openai-compatible", 
                        "name": "LM Studio", 
                        "options": {
                            "baseURL": f"{self.base_url}/v1",
                            "headers": {
                                "X-LM-Context-Length": str(context)
                            }
                        }, 
                        "models": {}
                    }
                }, 
                "agent": {}
            }
            think, code = None, None
            for m in models:
                mid = m['id']
                model_cfg = {"name": mid, "options": {"contextLength": context}}
                if "capabilities" in m: model_cfg["capabilities"] = m["capabilities"]
                mid_l = mid.lower()
                if ("coder" in mid_l or "thinking" in mid_l or "reasoning" in mid_l) and "tool_use" not in model_cfg.get("capabilities", []):
                    if "capabilities" not in model_cfg: model_cfg["capabilities"] = []
                    model_cfg["capabilities"].append("tool_use")
                cfg["provider"]["lmstudio"]["models"][mid] = model_cfg
                if not think and ("thinking" in mid_l or "reasoning" in mid_l): think = mid
                if not code and "coder" in mid_l: code = mid
            
            final_code = coder_id or code or (models[0]['id'] if models else None)
            final_think = thinking_id or think
            if final_code:
                cfg["model"] = f"lmstudio/{final_code}"
                print(f"Using Coder: {final_code}", file=sys.stderr)
            if final_think:
                cfg["agent"]["plan"] = {"model": f"lmstudio/{final_think}", "tools": {"write": False, "edit": False, "patch": False, "bash": False}}
                print(f"Using Planner: {final_think}", file=sys.stderr)
            print(json.dumps(cfg, indent=4))
        except Exception as e:
            print(f"Failed to generate OpenCode config: {e}", file=sys.stderr)

    def templates(self):
        t = {
            "Coder": "Expert software engineer. Concise code.",
            "Creative": "Creative writer. Evocative language.",
            "Logic": "Logical reasoning. Step-by-step.",
            "Summarizer": "Summarize text into bullets."
        }
        print("--- System Prompt Templates ---")
        for n, p in t.items(): print(f"[{n}]: {p}")

    def raw(self, method: str, endpoint: str, data: Optional[str] = None):
        try:
            resp = self._request(method, endpoint, json.loads(data) if data else None)
            print(json.dumps(resp, indent=4))
        except Exception as e:
            print(f"Raw request failed: {e}")

def main():
    p = argparse.ArgumentParser(description="Professional LM Studio CLI Utility")
    s = p.add_subparsers(dest="cmd", help="Available commands")
    
    conf = s.add_parser("config", help="Configure CLI settings")
    conf.add_argument("--url", help="Set the LM Studio base URL")
    conf.add_argument("--timeout", type=int, help="Set the request timeout in seconds")
    conf.add_argument("--vram", type=float, help="Set your GPU's VRAM in GB (for fit estimation)")
    conf.add_argument("--show", action="store_true", help="Show current configuration")
    
    stat = s.add_parser("status", help="Show server summary and loaded models")
    stat.add_argument("--watch", action="store_true", help="Auto-refresh status dashboard")
    
    s.add_parser("list", help="List all models in your local library")
    s.add_parser("models", help="Alias for 'list'")
    s.add_parser("check", help="Quick connectivity check to the server")
    
    load = s.add_parser("load", help="Load a model with optional settings")
    load.add_argument("model_id", help="The ID of the model to load")
    load.add_argument("--context", type=int, help="Set context length (e.g., 32768)")
    load.add_argument("--gpu", type=float, help="Set GPU offload ratio (0.0 to 1.0) or layer count")
    
    unl = s.add_parser("unload", help="Unload models from memory")
    unl.add_argument("model_id", nargs='?', help="The instance/model ID to unload")
    unl.add_argument("--all", action="store_true", help="Unload all currently loaded models")
    
    inf = s.add_parser("info", help="Get detailed model metadata (arch, quantization, etc.)")
    inf.add_argument("model_id", nargs='?', help="The ID of the model to inspect (optional, shows all loaded if omitted)")
    
    src = s.add_parser("search", help="Search local library and discover models on Hugging Face")
    src.add_argument("query", help="Search term (e.g., 'llama', 'coder')")
    
    dl = s.add_parser("download", help="Download a model from Hugging Face")
    dl.add_argument("model_id", help="Repo identifier (e.g., 'username/repo') or full URL")
    
    ds = s.add_parser("download-status", help="Check progress of background downloads")
    ds.add_argument("job_id", nargs='?', help="The job ID to track (optional)")
    
    s.add_parser("presets", help="List local LM Studio configuration presets")
    
    ch = s.add_parser("chat", help="Send a single chat message")
    ch.add_argument("model_id", help="The ID of the model to use")
    ch.add_argument("msg", help="The message to send")
    ch.add_argument("--system", help="Optional system prompt")
    ch.add_argument("--stream", action="store_true", help="Enable real-time token streaming")
    ch.add_argument("--temp", type=float, default=0.7, help="Inference temperature (default: 0.7)")
    ch.add_argument("--max-tokens", type=int, default=-1, help="Max tokens to generate")
    ch.add_argument("--top-p", type=float, default=1.0, help="Top P sampling (default: 1.0)")

    rep = s.add_parser("repl", help="Start an interactive streaming conversation")
    rep.add_argument("model_id", help="The ID of the model to use")
    rep.add_argument("--system", help="Optional system prompt")
    
    bn = s.add_parser("bench", help="Benchmark model performance (TTFT and TPS)")
    bn.add_argument("model_id", help="The ID of the model to benchmark")
    
    cp = s.add_parser("complete", help="Perform classic text completion (non-chat)")
    cp.add_argument("model_id", help="The ID of the model to use")
    cp.add_argument("prompt", help="The text prompt to complete")
    
    em = s.add_parser("embeddings", help="Generate vector embeddings for a text string")
    em.add_argument("model_id", help="The ID of the model to use")
    em.add_argument("input", help="The text to generate embeddings for")
    
    op = s.add_parser("opencode", help="Generate an OpenCode-compatible configuration")
    op.add_argument("--coder", help="Manually specify the coder model ID")
    op.add_argument("--think", help="Manually specify the thinking/planning model ID")
    op.add_argument("--context", type=int, default=32768, help="Context size for generated config (default: 32768)")
    
    s.add_parser("templates", help="Show useful system prompt templates")
    
    rw = s.add_parser("raw", help="Send a raw HTTP request to the API")
    rw.add_argument("method", choices=["GET", "POST"], help="HTTP method")
    rw.add_argument("endpoint", help="API endpoint (e.g., '/v1/models')")
    rw.add_argument("--data", help="Raw JSON data for POST requests")
    
    args = p.parse_args()
    m = ConfigManager()
    c = LMStudioClient(m.get("base_url"), m.get("timeout"), m.get("vram_gb"))
    
    if args.cmd == "config":
        if args.url: m.save_config("base_url", args.url)
        if args.timeout: m.save_config("timeout", args.timeout)
        if args.vram: m.save_config("vram_gb", args.vram)
        if args.show or (not args.url and not args.timeout and not args.vram): print(json.dumps(m.config, indent=4))
    elif args.cmd == "status": c.status(args.watch)
    elif args.cmd in ["list", "models"]: c.list_models()
    elif args.cmd == "check": c.check()
    elif args.cmd == "info": c.info(args.model_id)
    elif args.cmd == "load": c.load(args.model_id, args.context, args.gpu)
    elif args.cmd == "unload": c.unload(args.model_id, args.all)
    elif args.cmd == "search": c.search(args.query)
    elif args.cmd == "download": c.download(args.model_id)
    elif args.cmd == "download-status": c.download_status(args.job_id)
    elif args.cmd == "presets": c.presets()
    elif args.cmd == "chat": c.chat(args.model_id, args.msg, args.system, args.stream, args.temp, args.max_tokens, args.top_p)
    elif args.cmd == "repl": c.repl(args.model_id, args.system)
    elif args.cmd == "bench": c.bench(args.model_id)
    elif args.cmd == "complete": c.complete(args.model_id, args.prompt)
    elif args.cmd == "embeddings": c.embeddings(args.model_id, args.input)
    elif args.cmd == "opencode": c.opencode(args.coder, args.think, args.context)
    elif args.cmd == "templates": c.templates()
    elif args.cmd == "raw": c.raw(args.method, args.endpoint, args.data)
    else: p.print_help()

if __name__ == "__main__": main()
