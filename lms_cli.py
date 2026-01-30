#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

# Constants
CONFIG_DIR = os.path.expanduser("~/.config/lms-cli")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_CONFIG = {
    "base_url": "http://localhost:1234",
    "timeout": 30
}

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
    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, method=method, headers=headers)
        if data:
            req.data = json.dumps(data).encode('utf-8')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
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

    def status(self):
        print(f"--- LM Studio Status ---")
        print(f"Base URL: {self.base_url}")
        try:
            data = self._request("GET", "/api/v0/models")
            models = data.get('data', [])
            loaded_models = [m for m in models if m.get('state') == 'loaded']
            print(f"Status:   Online")
            print(f"Models:   {len(models)} available")
            if loaded_models:
                print(f"\nCurrently Loaded Models:")
                for m in loaded_models:
                    ctx = m.get('loaded_context_length', 'unknown')
                    print(f" - {m.get('id')} ({m.get('type')}, context: {ctx})")
            else:
                print("\nNo models currently loaded.")
            print("\nNote: System resources (CPU/GPU/RAM) are not exposed via API.")
        except Exception as e:
            print(f"Status:   Offline\nError:    {e}")

    def check(self):
        try:
            self._request("GET", "/v1/models")
            print(f"✔ LM Studio is reachable at {self.base_url}")
            return True
        except:
            print(f"✘ LM Studio is NOT reachable at {self.base_url}")
            return False

    def list_models(self):
        print(f"Available Models at {self.base_url}:")
        try:
            data = self._request("GET", "/v1/models")
            for model in data.get('data', []):
                print(f" - {model.get('id')}")
        except Exception as e:
            print(f"Error listing models: {e}")

    def info(self, model_id: str):
        try:
            data = self._request("GET", "/v1/models")
            model = next((m for m in data.get('data', []) if m['id'] == model_id), None)
            if model: print(json.dumps(model, indent=4))
            else: print(f"Model '{model_id}' not found.")
        except Exception as e: print(f"Error: {e}")

    def load(self, model_id: str, context: Optional[int] = None, gpu: Optional[float] = None):
        print(f"Loading {model_id}...")
        # LM Studio API expects settings in the root for the load endpoint
        payload = {"model": model_id}
        if context:
            payload["context_length"] = context
        if gpu is not None:
            # If 0.0-1.0 is provided, we might need to know total layers, 
            # but usually 'gpu_layers' expects an integer count. 
            # We'll treat the float as a ratio if it's <= 1.0, otherwise as a count.
            payload["gpu_layers"] = int(gpu) if gpu > 1 else -1 # -1 often means 'max'
            
        try:
            resp = self._request("POST", "/api/v1/models/load", payload)
            print(f"Successfully loaded: {resp.get('instance_id', model_id)}")
        except Exception as e:
            print(f"Load failed: {e}")
            # Fallback
            fb = {"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
            self._request("POST", "/v1/chat/completions", fb)
            print("Load triggered (fallback).")

    def unload(self, identifier: str):
        print(f"Unloading {identifier}...")
        try:
            # Newer LM Studio API expects 'instance_id'
            self._request("POST", "/api/v1/models/unload", {"instance_id": identifier})
            print("Unload successful.")
        except Exception:
            # Fallback for older versions which might use 'model' or 'id'
            try:
                self._request("POST", "/v1/models/unload", {"model": identifier})
                print("Unload successful (fallback).")
            except:
                print("Note: Unload failed. Ensure you are using the correct instance ID (see 'status').")

    def search(self, query: str):
        try:
            data = self._request("GET", "/v1/models")
            found = [m['id'] for m in data.get('data', []) if query.lower() in m['id'].lower()]
            if found:
                for m in found: print(f" - {m}")
            else: print("No local matches.")
        except Exception as e: print(f"Search failed: {e}")

    def download(self, model_id: str):
        try:
            resp = self._request("POST", "/api/v1/models/download", {"model": model_id})
            print(f"Download started: {json.dumps(resp, indent=2)}")
        except Exception as e: print(f"Failed: {e}")

    def chat(self, model_id: str, message: str, system: Optional[str] = None):
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": message})
        try:
            resp = self._request("POST", "/v1/chat/completions", {"model": model_id, "messages": msgs})
            print(f"\nResponse:\n{resp.get('choices')[0].get('message').get('content')}\n")
        except Exception as e: print(f"Chat failed: {e}")

    def complete(self, model_id: str, prompt: str):
        try:
            resp = self._request("POST", "/v1/completions", {"model": model_id, "prompt": prompt})
            print(f"\nResult:\n{resp.get('choices')[0].get('text')}\n")
        except Exception as e: print(f"Failed: {e}")

    def embeddings(self, model_id: str, input_text: str):
        try:
            resp = self._request("POST", "/v1/embeddings", {"model": model_id, "input": input_text})
            vec = resp.get('data')[0].get('embedding')
            print(f"Vector length: {len(vec)}\nFirst 5: {vec[:5]}")
        except Exception as e: print(f"Failed: {e}")

    def templates(self):
        t = {
            "Coder": "Expert software engineer. Concise code.",
            "Creative": "Creative writer. Evocative language.",
            "Logic": "Logical reasoning. Step-by-step.",
            "Summarizer": "Summarize text into bullets."
        }
        for n, p in t.items(): print(f"[{n}]: {p}")

    def opencode(self, coder_id: Optional[str] = None, thinking_id: Optional[str] = None):
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
                        "options": {"baseURL": f"{self.base_url}/v1"},
                        "models": {}
                    }
                },
                "agent": {}
            }
            
            detected_think, detected_code = None, None
            for m in models:
                mid = m['id']
                model_cfg = {"name": mid}
                
                # Copy existing capabilities (like tool_use) if present
                if "capabilities" in m:
                    model_cfg["capabilities"] = m["capabilities"]
                
                # Force tool_use for models known to support it if not already tagged
                mid_lower = mid.lower()
                is_coder = "coder" in mid_lower
                is_think = "thinking" in mid_lower or "reasoning" in mid_lower
                
                if (is_coder or is_think) and "tool_use" not in model_cfg.get("capabilities", []):
                    if "capabilities" not in model_cfg:
                        model_cfg["capabilities"] = []
                    model_cfg["capabilities"].append("tool_use")
                
                cfg["provider"]["lmstudio"]["models"][mid] = model_cfg
                
                if not detected_think and is_think:
                    detected_think = mid
                if not detected_code and is_coder:
                    if "7b" in mid_lower or "8b" in mid_lower:
                        detected_code = mid
            
            # Use manual overrides or detected models
            final_code = coder_id or detected_code or (models[0]['id'] if models else None)
            final_think = thinking_id or detected_think
            
            if final_code:
                cfg["model"] = f"lmstudio/{final_code}"
                print(f"Using Coder: {final_code}", file=sys.stderr)
            
            if final_think:
                cfg["agent"]["plan"] = {
                    "model": f"lmstudio/{final_think}",
                    "tools": {"write": False, "edit": False, "patch": False, "bash": False}
                }
                print(f"Using Planner: {final_think}", file=sys.stderr)
                
            print(json.dumps(cfg, indent=4))
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)

    def raw(self, method: str, endpoint: str, data: Optional[str] = None):
        try:
            resp = self._request(method, endpoint, json.loads(data) if data else None)
            print(json.dumps(resp, indent=4))
        except Exception as e: print(f"Failed: {e}")

def main():
    p = argparse.ArgumentParser(description="LM Studio CLI")
    s = p.add_subparsers(dest="cmd")
    
    conf = s.add_parser("config")
    conf.add_argument("--url")
    conf.add_argument("--timeout", type=int)
    conf.add_argument("--show", action="store_true")
    
    s.add_parser("status")
    s.add_parser("list")
    s.add_parser("models")
    s.add_parser("check")
    
    load = s.add_parser("load")
    load.add_argument("model_id")
    load.add_argument("--context", type=int)
    load.add_argument("--gpu", type=float)
    
    unl = s.add_parser("unload")
    unl.add_argument("model_id")
    
    inf = s.add_parser("info")
    inf.add_argument("model_id")
    
    src = s.add_parser("search")
    src.add_argument("query")
    
    dl = s.add_parser("download")
    dl.add_argument("model_id")
    
    ch = s.add_parser("chat")
    ch.add_argument("model_id")
    ch.add_argument("msg")
    ch.add_argument("--system")
    
    cp = s.add_parser("complete")
    cp.add_argument("model_id")
    cp.add_argument("prompt")
    
    em = s.add_parser("embeddings")
    em.add_argument("model_id")
    em.add_argument("input")
    
    op = s.add_parser("opencode", help="Generate OpenCode json config")
    op.add_argument("--coder", help="Manual override for coder model ID")
    op.add_argument("--think", help="Manual override for thinking model ID")
    
    s.add_parser("templates")
    
    rw = s.add_parser("raw")
    rw.add_argument("method", choices=["GET", "POST"])
    rw.add_argument("endpoint")
    rw.add_argument("--data")
    
    args = p.parse_args()
    m = ConfigManager()
    c = LMStudioClient(m.get("base_url"), m.get("timeout"))
    
    if args.cmd == "config":
        if args.url: m.save_config("base_url", args.url)
        if args.timeout: m.save_config("timeout", args.timeout)
        if args.show or (not args.url and not args.timeout): print(json.dumps(m.config, indent=4))
    elif args.cmd == "status": c.status()
    elif args.cmd in ["list", "models"]: c.list_models()
    elif args.cmd == "check": c.check()
    elif args.cmd == "info": c.info(args.model_id)
    elif args.cmd == "load": c.load(args.model_id, args.context, args.gpu)
    elif args.cmd == "unload": c.unload(args.model_id)
    elif args.cmd == "search": c.search(args.query)
    elif args.cmd == "download": c.download(args.model_id)
    elif args.cmd == "chat": c.chat(args.model_id, args.msg, args.system)
    elif args.cmd == "complete": c.complete(args.model_id, args.prompt)
    elif args.cmd == "embeddings": c.embeddings(args.model_id, args.input)
    elif args.cmd == "opencode": c.opencode(args.coder, args.think)
    elif args.cmd == "templates": c.templates()
    elif args.cmd == "raw": c.raw(args.method, args.endpoint, args.data)
    else: p.print_help()

if __name__ == "__main__": main()