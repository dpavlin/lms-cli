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
    "vram_gb": 12.0,
    "default_context": 32768
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
    def __init__(self, base_url: str, timeout: int, vram_gb: float = 12.0, default_context: int = 32768):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.vram_gb = vram_gb
        self.default_context = default_context

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

    def get_active_model(self) -> Optional[str]:
        try:
            data = self._request("GET", "/api/v0/models")
            for m in data.get('data', []):
                if m.get('state') == 'loaded':
                    return m['id']
        except:
            pass
        return None

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
                        print(f"\n--- Currently Loaded Models ({len(loaded)}) ---")
                        for m in loaded:
                            m_id = m.get('id')
                            
                            # Size/VRAM estimation
                            match = re.search(r'(\d+(?:\.\d+)?)[bB]', m_id)
                            size_str = "???"
                            vram_est = "???"
                            if match:
                                p = float(match.group(1))
                                size_str = f"{p:g}B"
                                est = (p * 0.6) + 1.5
                                vram_est = f"~{est:.1f}GB"

                            print(f"\nModel:        {m_id}")
                            print(f"  Architecture: {m.get('arch', 'N/A')}")
                            print(f"  Size:         {size_str} (Est. VRAM: {vram_est})")
                            print(f"  Quantization: {m.get('quantization', 'N/A')}")
                            print(f"  Context:      {m.get('loaded_context_length', 'N/A')} / {m.get('max_context_length', 'N/A')}")
                            print(f"  Capabilities: {', '.join(m.get('capabilities', [])) or 'None'}")
                    else:
                        print("\nNo models currently loaded.")
                except:
                    print(f"Status:   Offline")
                print("\nNote: System resources (CPU/GPU/RAM) are not exposed via API.")
                if not watch: break
                time.sleep(2)
        except KeyboardInterrupt: pass

    def list_models(self):
        vram_limit = self.vram_gb
        print(f"Available Models at {self.base_url}:")
        try:
            data = self._request("GET", "/api/v0/models")
            models = data.get('data', [])
            
            # Print header
            print(f" {'ID':<55} | {'Size':>5} | {'VRAM Est':>8} | {'GPU %':>6} | {'Quant':<8} | {'Capabilities'}")
            print(f" {'-'*55} | {'-'*5} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*20}")
            
            for model in models:
                m_id = model.get('id')
                quant = model.get('quantization', 'N/A')
                caps = ", ".join(model.get('capabilities', [])) or "None"
                is_loaded = model.get('state') == 'loaded'
                
                # Heuristic for Size and VRAM
                match = re.search(r'(\d+(?:\.\d+)?)[bB]', m_id)
                size_str = "???"
                vram_est = "???"
                gpu_pct_str = "  -  "
                
                if match:
                    params = float(match.group(1))
                    size_str = f"{params:g}B"
                    est_gb = (params * 0.6) + 1.5
                    vram_est = f"~{est_gb:.1f}GB"
                    
                    pct = (est_gb / vram_limit) * 100
                    gpu_pct_str = f"{int(pct)}%"
                    if is_loaded:
                        gpu_pct_str = f"*{gpu_pct_str}"
                
                print(f" {m_id:<55} | {size_str:>5} | {vram_est:>8} | {gpu_pct_str:>6} | {quant:<8} | {caps}")
            print(f"\nNote: GPU % is estimated against your {vram_limit}GB VRAM config.")
            print("      Asterisk (*) indicates the model is currently loaded in memory.")
        except Exception as e:
            print(f"Error listing models: {e}")

    def switch(self, query: Optional[str] = None):
        vram_limit = self.vram_gb
        try:
            data = self._request("GET", "/api/v0/models")
            all_models = data.get('data', [])
            
            # Filter models if query is provided
            if query:
                models = [m for m in all_models if query.lower() in m['id'].lower()]
                if not models:
                    print(f"No models matching '{query}' found.")
                    return
            else:
                models = all_models

            print(f"--- Switch Model (VRAM Limit: {vram_limit}GB) ---")
            print(f" # {'ID':<55} | {'Size':>5} | {'VRAM Est':>8} | {'GPU %':>6} | {'Quant':<8} | {'Capabilities'}")
            print(f" -{'-'*55}-|-------|----------|--------|----------|--------------------")
            
            current_vram_usage = 0
            
            for idx, model in enumerate(models):
                m_id = model.get('id')
                quant = model.get('quantization', 'N/A')
                is_loaded = model.get('state') == 'loaded'
                caps = ", ".join(model.get('capabilities', [])) or "None"
                
                match = re.search(r'(\d+(?:\.\d+)?)[bB]', m_id)
                size_str, vram_est, pct = "???", 0, 0
                if match:
                    p = float(match.group(1))
                    size_str = f"{p:g}B"
                    vram_est = (p * 0.6) + 1.5
                    pct = (vram_est / vram_limit) * 100
                
                if is_loaded:
                    current_vram_usage += vram_est
                
                gpu_str = f"{int(pct)}%" if pct > 0 else "???"
                if is_loaded: gpu_str = f"*{gpu_str}"
                
                status_icon = "🟢" if is_loaded else "  "
                vram_str = f"~{vram_est:.1f}GB" if vram_est > 0 else "???"
                
                # Apply bold if loaded
                line = f"{idx+1:>2} {m_id:<52} {status_icon} | {size_str:>5} | {vram_str:>8} | {gpu_str:>6} | {quant:<8} | {caps}"
                if is_loaded:
                    print(f"\033[1m{line}\033[0m")
                else:
                    print(line)

            print(f"\nCurrent Estimated VRAM Usage: {current_vram_usage:.1f}GB / {vram_limit}GB")
            
            choice = input("\nSelect model number to load (or Enter to cancel): ").strip()
            if not choice: return
            
            sel_idx = int(choice) - 1
            if not (0 <= sel_idx < len(models)):
                print("Invalid selection.")
                return
            
            target_model = models[sel_idx]
            t_id = target_model['id']
            
            # Always unload all models first
            self.unload(all_models=True)
            
            self.load(t_id)
            
            # Verification chat
            print(f"\nVerifying {t_id} is responding...")
            self.chat(t_id, "ping", max_tokens=10, stream=True)
            
        except (ValueError, KeyboardInterrupt, EOFError):
            pass
        except Exception as e:
            print(f"Switch failed: {e}")

    def load(self, model_id: str, context: Optional[int] = None, gpu: Optional[float] = None):
        ctx = context or self.default_context
        print(f"Loading {model_id} (context: {ctx})...", file=sys.stderr)
        payload = {"model": model_id, "context_length": ctx}
        if gpu is not None:
            payload["gpu_layers"] = int(gpu) if gpu > 1 else -1
        
        try:
            # Use a much longer timeout for loading (10 minutes)
            url = f"{self.base_url}/api/v1/models/load"
            headers = {'Content-Type': 'application/json'}
            req = urllib.request.Request(url, method="POST", headers=headers, data=json.dumps(payload).encode('utf-8'))
            with urllib.request.urlopen(req, timeout=600) as response:
                resp = json.loads(response.read().decode('utf-8'))
                print(f"Loaded instance: {resp.get('instance_id', model_id)}")
        except Exception as e:
            print(f"Primary load failed: {e}")
            print("Falling back to chat completion load...")
            # Use every known key to force context size in JIT
            fb_payload = {
                "model": model_id, 
                "messages": [{"role": "user", "content": "Hi"}], 
                "max_tokens": 1,
                "context_length": ctx,
                "num_ctx": ctx,
                "context_window": ctx,
                "contextLength": ctx
            }
            fb_headers = {'Content-Type': 'application/json', 'X-LM-Context-Length': str(ctx)}
            fb_req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", method="POST", headers=fb_headers, data=json.dumps(fb_payload).encode('utf-8'))
            try:
                with urllib.request.urlopen(fb_req, timeout=600) as response:
                    print("Load triggered (fallback).")
            except Exception as fe:
                print(f"Fallback failed: {fe}")

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

    def tool_test(self, model_id: str, log_file: Optional[str] = None):
        log_entries = []
        def log(msg, end="\n", flush=True, terminal=True):
            if terminal:
                print(msg, end=end, flush=flush)
            log_entries.append(msg + end)

        log(f"--- Comprehensive Tool Test: {model_id} ---")
        log(f"Started at: {time.ctime()}\n")
        
        tools = [
            {"type": "function", "function": {"name": "list_files", "description": "Lists files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "Reads a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Creates a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "patch_file", "description": "Updates a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "search": {"type": "string"}, "replace": {"type": "string"}}, "required": ["path", "search", "replace"]}}},
            {"type": "function", "function": {"name": "search_file_content", "description": "Greps content", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}}},
            {"type": "function", "function": {"name": "run_shell_command", "description": "Execbash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
        ]

        individual_tests = [
            ("list_files", "List files in the current directory."),
            ("read_file", "Read the content of 'lms_cli.py'."),
            ("write_file", "Create a new file 'test.log' with 'log entry'."),
            ("patch_file", "In 'config.py', change 'DEBUG = False' to 'DEBUG = True'."),
            ("search_file_content", "Search for the string 'import' inside 'lms_cli.py'."),
            ("run_shell_command", "Run the command 'uname -a' in the shell.")
        ]

        results = {}
        total_start = time.time()

        log("STAGE 1: Individual Tool Tests")
        for tool_name, prompt in individual_tests:
            log(f"  Testing {tool_name:<20} ... ", end="")
            step_start = time.time()
            payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "tools": tools, "tool_choice": "auto"}
            log(f"\n[REQUEST PAYLOAD]:\n{json.dumps(payload, indent=2)}", terminal=False)
            try:
                resp = self._request("POST", "/v1/chat/completions", payload)
                log(f"\n[RESPONSE JSON]:\n{json.dumps(resp, indent=2)}", terminal=False)
                tc = resp.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
                duration = time.time() - step_start
                if any(call['function']['name'] == tool_name for call in tc):
                    log(f"✅ ({duration:.2f}s)")
                    results[tool_name] = True
                else:
                    log(f"❌ ({duration:.2f}s)")
                    results[tool_name] = False
            except Exception as e:
                log(f"⚠️ Error: {e}")
                results[tool_name] = False

        log("\nSTAGE 2: Multi-Tool (Parallel) Tests")
        multi_prompts = [
            ("Direct", "Do all at once: list files in '.', search for 'TODO' in 'lms_cli.py', and run 'date'."),
            ("Strict", "SYSTEM: STRICTOR TEST. You MUST output 3 tool calls now: 1. list_files('.'), 2. run_shell_command('whoami'), 3. write_file('notes.txt', 'hello').")
        ]

        for label, prompt in multi_prompts:
            log(f"  Testing Multi-Tool ({label:<6}) ... ", end="")
            step_start = time.time()
            payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "tools": tools, "tool_choice": "auto"}
            log(f"\n[REQUEST PAYLOAD]:\n{json.dumps(payload, indent=2)}", terminal=False)
            try:
                resp = self._request("POST", "/v1/chat/completions", payload)
                log(f"\n[RESPONSE JSON]:\n{json.dumps(resp, indent=2)}", terminal=False)
                tc = resp.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
                duration = time.time() - step_start
                if len(tc) >= 3:
                    log(f"✅ ({len(tc)} tools, {duration:.2f}s)")
                    results[f"multi_{label}"] = True
                elif len(tc) > 0:
                    log(f"🟡 ({len(tc)} tools, {duration:.2f}s)")
                    results[f"multi_{label}"] = False
                else:
                    log(f"❌ ({duration:.2f}s)")
                    results[f"multi_{label}"] = False
            except Exception as e:
                log(f"⚠️ Error: {e}")
                results[f"multi_{label}"] = False

        total_duration = time.time() - total_start
        log(f"\n--- Summary for {model_id} ---")
        log(f"Total time: {total_duration:.2f}s")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        log(f"Score: {passed}/{total}")
        
        if passed == total:
            log("Verdict: 🚀 EXCELLENT. Fully capable agent model.")
        elif results.get("multi_Direct") or results.get("multi_Strict"):
            log("Verdict: 🟢 GOOD. Supports parallel tool calls.")
        elif any(results.values()):
            log("Verdict: 🟡 LIMITED. Single-tool only. OpenCode will be slow.")
        else:
            log("Verdict: 🔴 INCAPABLE. Does not support tools via API.")

        if log_file:
            try:
                with open(log_file, 'w') as f:
                    f.writelines(log_entries)
                print(f"\nDetailed log saved to: {log_file}")
            except Exception as e:
                print(f"\nFailed to save log: {e}")

    def download(self, model_id: str, auto_switch: bool = True):
        original_input = model_id
        if not model_id.startswith("http") and "/" in model_id:
            print(f"Converting '{model_id}' to Hugging Face URL...")
            model_id = f"https://huggingface.co/{model_id}"
        
        print(f"Requesting download for: {model_id}")
        try:
            resp = self._request("POST", "/api/v1/models/download", {"model": model_id})
            job_id = resp.get('job_id')
            if not job_id:
                print(f"Download started: {json.dumps(resp, indent=2)}")
                return
            
            print(f"Download started (Job ID: {job_id})")
            
            # Monitoring loop
            success = False
            try:
                while True:
                    status = self.download_status(job_id, internal=True)
                    if not status:
                        print("\nLost track of download job.")
                        break
                    
                    state = status.get('status', 'unknown')
                    total = status.get('total_size_bytes', 0)
                    cur = status.get('downloaded_bytes', 0)
                    speed = status.get('bytes_per_second', 0) / 1024 / 1024
                    
                    prog = (cur / total * 100) if total > 0 else 0
                    
                    print(f"\rProgress: [{'#'*int(prog/5):<20}] {prog:>3.0f}% | {speed:>6.1f} MB/s | {state}", end="", flush=True)
                    
                    if state == 'completed':
                        print("\n✅ Download finished successfully.")
                        success = True
                        break
                    if state == 'failed':
                        print(f"\n❌ Download failed: {status.get('error', 'Unknown error')}")
                        break
                    
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\nStopped monitoring. Download continues in background (Job: {job_id})")
            
            if success and auto_switch:
                print(f"\nSwitching to {original_input}...")
                # Give LM Studio more time to index the new file
                time.sleep(5)
                
                resolved_id = None
                try:
                    data = self._request("GET", "/api/v0/models")
                    all_m = data.get('data', [])
                    
                    # Try matching strategies
                    # 1. Exact match (case insensitive)
                    for m in all_m:
                        if m['id'].lower() == original_input.lower():
                            resolved_id = m['id']
                            break
                    
                    # 2. Contains match
                    if not resolved_id:
                        for m in all_m:
                            if original_input.lower() in m['id'].lower() or m['id'].lower() in original_input.lower():
                                resolved_id = m['id']
                                break
                    
                    # 3. Basename match
                    if not resolved_id:
                        base = original_input.split("/")[-1].lower()
                        for m in all_m:
                            if base in m['id'].lower():
                                resolved_id = m['id']
                                break
                except: pass
                
                final_id = resolved_id or original_input
                self.unload(all_models=True)
                self.load(final_id)
                print(f"\nVerifying {final_id} is responding...")
                self.chat(final_id, "ping", max_tokens=10, stream=True)
                
        except Exception as e:
            print(f"Download failed: {e}")

    def download_status(self, job_id: str = None, internal: bool = False):
        endpoint = "/api/v1/models/download/status"
        if job_id: endpoint += f"/{job_id}"
        try:
            data = self._request("GET", endpoint)
            if internal: return data
            print("Download Status:")
            print(json.dumps(data, indent=4))
        except:
            if internal: return None
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
        print(f" (Hiding models > {vram_limit}GB VRAM)")
        encoded_query = urllib.parse.quote(query)
        hf_url = f"https://huggingface.co/api/models?search={encoded_query}&filter=gguf&sort=downloads&direction=-1&limit=100"
        try:
            req = urllib.request.Request(hf_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                models = json.loads(response.read().decode('utf-8'))
                if not models:
                    print("No models found on Hugging Face.")
                    return
                
                count = 0
                displayed_models = []
                for m in models:
                    if count >= 20: break
                    
                    m_id = m.get('modelId')
                    downloads = m.get('downloads', 0)
                    tags = m.get('tags', [])
                    match = re.search(r'(\d+)b', m_id.lower())
                    vram_est = "???"
                    fit_icon = "⚪"
                    
                    # Tool support heuristic
                    has_tools = False
                    if any(t in tags for t in ["tool-use", "function-calling"]):
                        has_tools = True
                    else:
                        m_id_lower = m_id.lower()
                        if any(k in m_id_lower for k in ["instruct", "coder", "thinking", "reasoning"]):
                            has_tools = True
                    
                    tool_icon = "🛠️" if has_tools else "  "
                    
                    if match:
                        params = int(match.group(1))
                        est_gb = (params * 0.6) + 1.5
                        if est_gb > vram_limit * 1.1: # Skip if clearly too big
                            continue
                        
                        vram_est = f"~{est_gb:.1f}GB"
                        if est_gb <= vram_limit: fit_icon = "🟢"
                        else: fit_icon = "🟡"
                    
                    count += 1
                    displayed_models.append(m_id)
                    print(f"[{count:2}] {fit_icon} {m_id:<55} | {tool_icon} | {vram_est:>7} | ↓ {downloads}")
                
                print(f"\nShown {count} models that fit your hardware.")
                print("Legend: 🟢 Fits  🟡 Tight  ⚪ Unknown size  🛠️ Tool Support")
                
                if sys.stdin.isatty():
                    try:
                        choice = input("\nEnter number to download (or Enter to skip): ").strip()
                        if choice:
                            idx = int(choice) - 1
                            if 0 <= idx < len(displayed_models):
                                self.download(displayed_models[idx])
                            else:
                                print("Invalid number.")
                    except (ValueError, KeyboardInterrupt, EOFError):
                        pass
        except Exception as e:
            print(f"HF search failed: {e}")

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
            
            # GPU Detection Heuristic
            status = "❌ CPU (No Acceleration)"
            if tps > 15:
                status = "✅ GPU (Full Acceleration)"
            elif tps > 5:
                status = "🟡 Hybrid (Partial Offload)"
            
            print(f"\nResults for {model_id}:")
            print(f"  Speed:  {tps:.2f} tokens/sec")
            print(f"  TTFT:   {ttft:.4f}s")
            print(f"  Tokens: {token_count}")
            print(f"  Status: {status}")
            
            if "CPU" in status:
                print("\n  Hint: If this should be on GPU, try reloading with:")
                print(f"  ./lms_cli.py load {model_id} --gpu 1.0")
                
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

    def opencode(self, coder_id: Optional[str] = None, thinking_id: Optional[str] = None, context: Optional[int] = None):
        ctx_size = context or self.default_context
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
                                "X-LM-Context-Length": str(ctx_size)
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
                model_cfg = {"name": mid, "options": {"contextLength": ctx_size}}
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
                

    def top(self):
        try:
            while True:
                start_req = time.time()
                try:
                    # 1. Gather Data
                    data = self._request("GET", "/api/v0/models")
                    models = data.get('data', [])
                    loaded = [m for m in models if m.get('state') == 'loaded']
                    
                    downloads = []
                    try:
                        downloads = self._request("GET", "/api/v1/models/download/status")
                    except: pass
                    
                    latency = (time.time() - start_req) * 1000
                    
                    # 2. Render UI
                    os.system('clear')
                    print(f"LM STUDIO TOP | {self.base_url} | Latency: {latency:.0f}ms")
                    print(f"VRAM Config: {self.vram_gb}GB | Status: Online")
                    print("-" * 80)
                    
                    # Loaded Models Section
                    print(f"\nLOADED MODELS ({len(loaded)}):")
                    if loaded:
                        print(f" {'ID':<50} | {'VRAM':>8} | {'GPU %':>6} | {'Ctx':>8}")
                        for m in loaded:
                            mid = m['id']
                            ctx = m.get('loaded_context_length', '???')
                            
                            # VRAM Est logic
                            match = re.search(r'(\d+(?:\.\d+)?)[bB]', mid)
                            vram_str, gpu_str = "???", "???"
                            if match:
                                p = float(match.group(1))
                                est = (p * 0.6) + 1.5
                                vram_str = f"~{est:.1f}GB"
                                gpu_str = f"{int((est/self.vram_gb)*100)}%"
                            
                            print(f" 🟢 {mid:<47} | {vram_str:>8} | {gpu_str:>6} | {ctx:>8}")
                    else:
                        print(" (No models loaded)")
                        
                    # Downloads Section
                    if downloads:
                        print(f"\nACTIVE DOWNLOADS ({len(downloads)}):")
                        for d in downloads:
                            jid = d.get('job_id', '???')
                            status = d.get('status', '???')
                            prog = 0
                            if d.get('total_size_bytes'):
                                prog = (d['downloaded_bytes'] / d['total_size_bytes']) * 100
                            
                            speed = d.get('bytes_per_second', 0) / 1024 / 1024
                            print(f" {jid:<15} | [{'#'*int(prog/5):<20}] {prog:>3.0f}% | {speed:>6.1f} MB/s | {status}")
                    
                    print(f"\n(Press Ctrl+C to exit)")
                    
                except Exception as e:
                    os.system('clear')
                    print(f"LM STUDIO TOP | Status: OFFLINE")
                    print(f"Error: {e}")
                
                time.sleep(2)
        except KeyboardInterrupt:
            pass

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
    conf.add_argument("--context", type=int, help="Set default context length for loading models")
    conf.add_argument("--show", action="store_true", help="Show current configuration")
    
    stat = s.add_parser("status", help="Show server summary and loaded models")
    stat.add_argument("--watch", action="store_true", help="Auto-refresh status dashboard")
    
    s.add_parser("list", help="List all models in your local library")
    s.add_parser("models", help="Alias for 'list'")
    sw = s.add_parser("switch", help="Interactively select a model to load, with VRAM management")
    sw.add_argument("query", nargs='?', help="Optional partial model name to filter the list")
    s.add_parser("check", help="Quick connectivity check to the server")
    
    load = s.add_parser("load", help="Load a model with optional settings")
    load.add_argument("model_id", help="The ID of the model to load")
    load.add_argument("--context", type=int, help="Set context length (overrides default)")
    load.add_argument("--gpu", type=float, help="Set GPU offload ratio (0.0 to 1.0) or layer count")
    
    unl = s.add_parser("unload", help="Unload models from memory")
    unl.add_argument("model_id", nargs='?', help="The instance/model ID to unload")
    unl.add_argument("--all", action="store_true", help="Unload all currently loaded models")
    
    tt = s.add_parser("tool-test", help="Verify if a model supports tool calling")
    tt.add_argument("model_id", nargs='?', help="The ID of the model to use (optional if model loaded)")
    tt.add_argument("--log", help="Path to save a detailed test log")
    
    src = s.add_parser("search", help="Search local library and discover models on Hugging Face")
    src.add_argument("query", help="Search term (e.g., 'llama', 'coder')")
    
    dl = s.add_parser("download", help="Download a model from Hugging Face")
    dl.add_argument("model_id", help="Repo identifier (e.g., 'username/repo') or full URL")
    dl.add_argument("--no-switch", action="store_true", help="Do not automatically switch to the model after download")
    
    ds = s.add_parser("download-status", help="Check progress of background downloads")
    ds.add_argument("job_id", nargs='?', help="The job ID to track (optional)")
    
    s.add_parser("presets", help="List local LM Studio configuration presets")
    
    ch = s.add_parser("chat", help="Send a single chat message")
    ch.add_argument("model_id", nargs='?', help="The ID of the model to use (optional if model loaded)")
    ch.add_argument("msg", help="The message to send")
    ch.add_argument("--system", help="Optional system prompt")
    ch.add_argument("--stream", action="store_true", help="Enable real-time token streaming")
    ch.add_argument("--temp", type=float, default=0.7, help="Inference temperature (default: 0.7)")
    ch.add_argument("--max-tokens", type=int, default=-1, help="Max tokens to generate")
    ch.add_argument("--top-p", type=float, default=1.0, help="Top P sampling (default: 1.0)")

    rep = s.add_parser("repl", help="Start an interactive streaming conversation")
    rep.add_argument("model_id", nargs='?', help="The ID of the model to use (optional if model loaded)")
    rep.add_argument("--system", help="Optional system prompt")
    
    bn = s.add_parser("bench", help="Benchmark model performance (TTFT and TPS)")
    bn.add_argument("model_id", nargs='?', help="The ID of the model to benchmark (optional if model loaded)")
    
    cp = s.add_parser("complete", help="Perform classic text completion (non-chat)")
    cp.add_argument("model_id", nargs='?', help="The ID of the model to use (optional if model loaded)")
    cp.add_argument("prompt", help="The text prompt to complete")
    
    em = s.add_parser("embeddings", help="Generate vector embeddings for a text string")
    em.add_argument("model_id", nargs='?', help="The ID of the model to use (optional if model loaded)")
    em.add_argument("input", help="The text to generate embeddings for")
    
    op = s.add_parser("opencode", help="Generate OpenCode json config")
    op.add_argument("--coder", help="Manual override for coder model ID")
    op.add_argument("--think", help="Manual override for thinking model ID")
    op.add_argument("--context", type=int, help="Context size for generated config (overrides default)")
    
    s.add_parser("top", help="Real-time top-like dashboard of LM Studio server")
    s.add_parser("templates", help="Show useful system prompt templates")
    
    rw = s.add_parser("raw", help="Send a raw HTTP request to the API")
    rw.add_argument("method", choices=["GET", "POST"], help="HTTP method")
    rw.add_argument("endpoint", help="API endpoint (e.g., '/v1/models')")
    rw.add_argument("--data", help="Raw JSON data for POST requests")
    
    args = p.parse_args()
    m = ConfigManager()
    c = LMStudioClient(m.get("base_url"), m.get("timeout"), m.get("vram_gb"), m.get("default_context"))
    
    def resolve_model(m_id):
        if m_id: return m_id
        active = c.get_active_model()
        if active:
            print(f"Using active model: {active}", file=sys.stderr)
            return active
        print("Error: No model specified and no model is currently loaded.", file=sys.stderr)
        sys.exit(1)

    if args.cmd == "config":
        if args.url: m.save_config("base_url", args.url)
        if args.timeout: m.save_config("timeout", args.timeout)
        if args.vram: m.save_config("vram_gb", args.vram)
        if args.context: m.save_config("default_context", args.context)
        if args.show or (not args.url and not args.timeout and not args.vram and not args.context): print(json.dumps(m.config, indent=4))
    elif args.cmd == "status": c.status(args.watch)
    elif args.cmd in ["list", "models"]: c.list_models()
    elif args.cmd == "switch": c.switch(args.query)
    elif args.cmd == "check": c.check()
    elif args.cmd == "tool-test": c.tool_test(resolve_model(args.model_id), args.log)
    elif args.cmd == "load": c.load(args.model_id, args.context, args.gpu)
    elif args.cmd == "unload": 
        target = args.model_id
        if not target and not args.all:
            target = c.get_active_model()
            if not target:
                print("Error: No model specified and no model is currently loaded.", file=sys.stderr)
                sys.exit(1)
            print(f"Using active model: {target}", file=sys.stderr)
        c.unload(target, args.all)
    elif args.cmd == "search": c.search(args.query)
    elif args.cmd == "download": c.download(args.model_id, not args.no_switch)
    elif args.cmd == "download-status": c.download_status(args.job_id)
    elif args.cmd == "presets": c.presets()
    elif args.cmd == "chat": c.chat(resolve_model(args.model_id), args.msg, args.system, args.stream, args.temp, args.max_tokens, args.top_p)
    elif args.cmd == "repl": c.repl(resolve_model(args.model_id), args.system)
    elif args.cmd == "bench": c.bench(resolve_model(args.model_id))
    elif args.cmd == "complete": c.complete(resolve_model(args.model_id), args.prompt)
    elif args.cmd == "embeddings": c.embeddings(resolve_model(args.model_id), args.input)
    elif args.cmd == "opencode": c.opencode(args.coder, args.think, args.context)
    elif args.cmd == "top": c.top()
    elif args.cmd == "templates": c.templates()
    elif args.cmd == "raw": c.raw(args.method, args.endpoint, args.data)
    else: p.print_help()

if __name__ == "__main__": main()
