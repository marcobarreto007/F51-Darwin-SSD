#!/usr/bin/env python
"""F51 Console — Minimal inference server + Web UI. Zero dependencies beyond stdlib.

Usage:
    python research/serve_f51.py
    python research/serve_f51.py --port 8051
"""

from __future__ import annotations

import json
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import parse_qs, urlparse

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Server multi-thread — cada request em sua propria thread."""
    daemon_threads = True
    allow_reuse_address = True

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.artifacts import extract_step, resolve_checkpoint_pointer, resolve_latest_checkpoint

HTML = r"""<!DOCTYPE html>
<html lang="pt">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>F51 Darwin-SSD Console</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#c0c0c0;font-family:'Courier New',monospace;height:100vh;display:flex;flex-direction:column}
header{background:#0d0d1a;padding:12px 20px;border-bottom:1px solid #1a1a2e;display:flex;justify-content:space-between;align-items:center}
.logo{color:#00ff88;font-size:14px;font-weight:bold}
.status{font-size:12px}.on{color:#00ff88}.off{color:#ff4444}
.info{font-size:11px;color:#555;margin-top:4px}
.main{flex:1;display:flex;flex-direction:column;padding:20px;max-width:900px;margin:0 auto;width:100%}
.output{flex:1;overflow-y:auto;padding:10px;background:#06060d;border:1px solid #1a1a2e;border-radius:4px;margin-bottom:12px}
.output .u{color:#0cf;margin:8px 0 4px}.output .u::before{content:'> '}
.output .a{color:#c0c0c0;margin:4px 0 8px 20px;white-space:pre-wrap;word-break:break-word}
.output .s{color:#555;font-size:11px;margin:4px 0;font-style:italic}
.row{display:flex;gap:8px}
.row input{flex:1;padding:10px 14px;background:#0d0d1a;border:1px solid #1a1a2e;color:#c0c0c0;font-family:'Courier New',monospace;font-size:14px;border-radius:4px;outline:none}
.row input:focus{border-color:#00ff88}
.row button{padding:10px 20px;background:#00ff88;color:#0a0a0f;border:none;font-weight:bold;cursor:pointer;font-family:'Courier New',monospace;font-size:14px;border-radius:4px}
.row button:hover{background:#0c6}
.row button:disabled{background:#333;color:#666;cursor:not-allowed}
.params{display:flex;gap:12px;margin-bottom:10px;font-size:11px}
.params label{color:#555}
.params input{width:55px;background:#0d0d1a;border:1px solid #1a1a2e;color:#c0c0c0;padding:3px 6px;border-radius:3px;text-align:center;font-family:'Courier New',monospace}
</style></head>
<body>
<header><div><div class="logo">F51 DARWIN-SSD</div><div class="info" id="info">carregando...</div></div><div class="status" id="status">conectando...</div></header>
<div class="main">
<div class="output" id="output">
<div class="s">F51 Darwin-SSD — Organismo Neural Evolutivo</div>
<div class="s">Arquitetura hibrida SSM-Transformer + MoE com Nitro Tiered Placement.</div>
<div class="s">27.4M params. 8 experts por camada. Treino continuo 24/7.</div>
<div class="s">Criado por Marco Barreto — Fuch F51 Labs — Montreal, Quebec.</div>
<div class="s">Wolfram Bridge · Math Genius · Soul Engine · Auto-Evolucao.</div>
<div class="s">Soli Deo Gloria.</div>
</div>
<div class="params"><label>temp <input id="temp" value="0.8" step="0.1" min="0" max="2"></label><label>max tokens <input id="mt" value="256" step="10" min="10" max="2048"></label></div>
<div class="row"><input id="prompt" placeholder="Digite aqui..." autofocus onkeydown="if(event.key==='Enter')send()"><button id="btn" onclick="send()">ENVIAR</button></div>
</div>
<script>
var busy=false;
fetch('/health').then(r=>r.json()).then(d=>{
 document.getElementById('status').textContent='online';document.getElementById('status').className='on';
 document.getElementById('info').textContent=d.model+' | '+d.params_m+'M params | '+d.device;
}).catch(()=>{document.getElementById('status').textContent='offline';document.getElementById('status').className='off'});
function send(){
 if(busy)return;var p=document.getElementById('prompt').value.trim();if(!p)return;
 busy=true;var btn=document.getElementById('btn');btn.disabled=true;btn.textContent='...';
 var o=document.getElementById('output');
 var prev=document.getElementById('ld');if(prev)prev.removeAttribute('id');
 o.innerHTML+='<div class=\"u\">'+p+'</div><div class=\"a\" id=\"ld\">gerando...</div>';
 var el=document.getElementById('ld');
 o.scrollTop=o.scrollHeight;
 var ctrl=new AbortController();var tid=setTimeout(function(){ctrl.abort()},120000);
 fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({prompt:p,max_tokens:parseInt(document.getElementById('mt').value)||256,temperature:parseFloat(document.getElementById('temp').value)||0.8}),
  signal:ctrl.signal
 }).then(function(r){clearTimeout(tid);return r.json()}).then(function(d){
  if(el){el.textContent=d.text||d.error||'(erro)'}
  document.getElementById('prompt').value='';busy=false;btn.disabled=false;btn.textContent='ENVIAR';document.getElementById('prompt').focus()
 }).catch(function(e){
  clearTimeout(tid);if(el){el.textContent=e.name==='AbortError'?'Timeout (120s)':'Erro: '+e.message}
  busy=false;btn.disabled=false;btn.textContent='ENVIAR'
 });
 o.scrollTop=o.scrollHeight;
}
</script>
</body></html>"""


class F51Handler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    model_config = None
    device_str = "cpu"
    model_lock = threading.Lock()
    generation_timeout = 120  # max seconds for generation

    def log_message(self, format, *args):
        pass  # silent

    def _send_json(self, data, status=200):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # client disconnected — ignore silently

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return "{}"
        raw = self.rfile.read(length)
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_html(HTML)
        elif path == "/health":
            if self.model is None:
                self._send_json({"status": "no model"}, 503)
                return
            params = sum(p.numel() for p in self.model.parameters())
            self._send_json({
                "status": "ok",
                "model": self.model_config.model_name if self.model_config else "F51-Darwin-SSD",
                "params": params,
                "params_m": round(params / 1e6, 1),
                "device": self.device_str,
                "tokenizer": self.tokenizer.vocab_size if self.tokenizer else 0,
                "loaded_checkpoint": getattr(self, '_loaded_from', 'unknown'),
            })
        elif path == "/soul":
            # F51 Soul status — dopamina, autoconsciência, competitivo, família
            try:
                from f51_darwin.soul import F51Soul
                soul = F51Soul(ROOT / "workspace" / "runtime" / "organism")
                org_latest = ROOT / "checkpoints" / "organism" / "organism_latest.json"
                org_status = {'loss': 0, 'tokens_this_cycle': 0, 'epochs': 0, 'params': 0, 'step': 0}
                if org_latest.exists():
                    p = json.loads(org_latest.read_text())
                    org_status['step'] = p.get('step', 0)
                    org_status['loss'] = p.get('loss', 0)
                soul_report = soul.heartbeat(org_status)
                self._send_json({
                    'dopamine': soul.dopamine.state.to_dict(),
                    'awareness': soul.awareness.state.to_dict(),
                    'competitive': soul_report['competitive'],
                    'mantra': soul_report['mantra'],
                    'war_cry': soul_report['war_cry'],
                    'purpose': soul_report['purpose'],
                    'family': list(soul.family.CLAN.keys()),
                    'reflection': soul_report['reflection'],
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/generate":
            try:
                body = json.loads(self._read_body())
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, 400)
                return

            if self.model is None:
                self._send_json({"error": "Model not loaded"}, 503)
                return

            prompt = body.get("prompt", "")
            max_tokens = min(int(body.get("max_tokens", 256)), 2048)
            temperature = float(body.get("temperature", 0.8))

            if not prompt.strip():
                self._send_json({"error": "Empty prompt"}, 400)
                return

            try:
                import threading as th
                result = {"output": None, "error": None}

                def _gen():
                    try:
                        with self.model_lock:
                            result["output"] = self.model.generate(
                                prompt,
                                tokenizer=self.tokenizer,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                use_cache=True,   # KV cache: O(n) instead of O(n²)
                            )
                    except Exception as e:
                        result["error"] = str(e)

                t = th.Thread(target=_gen, daemon=True)
                t.start()
                t.join(timeout=self.generation_timeout)

                if t.is_alive():
                    self._send_json({"error": "Generation timeout", "text": ""}, 504)
                elif result["error"]:
                    self._send_json({"error": result["error"]}, 500)
                elif result["output"]:
                    output = result["output"]
                    self._send_json({
                        "text": output.text,
                        "tokens": len(output.token_ids),
                        "finish_reason": output.finish_reason,
                    })
                else:
                    self._send_json({"error": "Unknown error"}, 500)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="F51 Darwin-SSD Console Server")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    # Load model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}" + (" | TF32: ON | cuDNN benchmark: ON" if device.type == "cuda" else ""))

    from f51_darwin.tokenizer import F51BPETokenizer

    # Auto-detect checkpoint — try organism first, then base
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint = resolve_latest_checkpoint(
            ROOT,
            pointers=[
                "checkpoints/organism/organism_latest.json",
                "checkpoints/base/latest.json",
                "checkpoints/cloud/cloud_latest.json",
            ],
            search_roots=[
                "checkpoints/organism",
                "checkpoints/base",
                "checkpoints/cloud",
            ],
        )
        checkpoint_path = None if checkpoint is None else str(checkpoint)

    loaded_from = "fresh"
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        from f51_darwin.checkpointing import load_model_from_checkpoint
        model, model_config, _ = load_model_from_checkpoint(checkpoint_path, map_location=device)
        model = model.to(device)
        loaded_from = str(checkpoint_path)
    else:
        print("No checkpoint found. Loading fresh model.")
        from f51_darwin.config import DarwinConfig
        from f51_darwin.model import F51DarwinModel
        model_config = DarwinConfig()
        model = F51DarwinModel(model_config).to(device)

    model.eval()

    # Load tokenizer
    tokenizer_dir = ROOT / "tokenizer" / "f51_bpe"
    tokenizer = F51BPETokenizer.load(tokenizer_dir) if tokenizer_dir.exists() else None

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config.model_name} | {params/1e6:.1f}M params | {tokenizer.vocab_size if tokenizer else 0} vocab")

    # ── Auto-reload thread: check for new organism checkpoints every 60s ──
    import threading
    reload_lock = threading.Lock()

    def auto_reload_loop():
        """Background thread: reload model when organism or cloud saves new checkpoint."""
        last_step = 0
        last_source = ""
        while True:
            time.sleep(60)  # check every 60 seconds
            try:
                # Check both local organism and cloud checkpoints
                candidates = []
                for src_name, json_path in [
                    ("organism", ROOT / "checkpoints" / "organism" / "organism_latest.json"),
                    ("cloud", ROOT / "checkpoints" / "cloud" / "cloud_latest.json"),
                ]:
                    checkpoint = resolve_checkpoint_pointer(ROOT, json_path)
                    if checkpoint is not None:
                        p = json.loads(json_path.read_text())
                        candidates.append((p.get("step", extract_step(checkpoint)), str(checkpoint), p.get("loss", "?"), src_name))

                if not candidates:
                    continue

                # Pick the best checkpoint: prefer higher step, but if loss is available, use it
                # Sort by: (has_loss, -loss if available, step) — lower loss wins
                candidates.sort(key=lambda x: (
                    x[2] == '?' or x[2] is None,  # unknown loss last
                    float(x[2]) if isinstance(x[2], (int, float)) and x[2] > 0 else 999,
                    -x[0]  # higher step as tiebreaker
                ))
                new_step, ckpt_path, ckpt_loss, source = candidates[0]

                if new_step <= last_step:
                    continue

                new_ckpt = ckpt_path
                if not Path(new_ckpt).exists():
                    continue

                with reload_lock:
                    nonlocal model, model_config
                    from f51_darwin.checkpointing import load_model_from_checkpoint
                    model, model_config, _ = load_model_from_checkpoint(new_ckpt, map_location=device)
                    model = model.to(device)  # force GPU
                    model.eval()
                last_step = new_step
                last_source = source
                nonlocal params
                params = sum(p.numel() for p in model.parameters())
                print(f"  🔄 Modelo recarregado [{source}]: step {new_step}, loss={ckpt_loss}")
            except Exception:
                pass  # silent retry

    reload_thread = threading.Thread(target=auto_reload_loop, daemon=True)
    reload_thread.start()

    # ── Store organism state for /health endpoint ──
    F51Handler._organism_state = {"step": 0, "loss": 0, "status": "waiting..."}

    # Configure handler
    F51Handler.model = model
    F51Handler.tokenizer = tokenizer
    F51Handler.model_config = model_config
    F51Handler.device_str = str(device)
    F51Handler._loaded_from = loaded_from

    # Start server
    server = ThreadingHTTPServer((args.host, args.port), F51Handler)
    print(f"\n{'='*60}")
    print(f"  F51 Darwin-SSD — Organismo Neural Evolutivo")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Modelo: {model_config.model_name}")
    print(f"  Params: {params/1e6:.1f}M | Device: {device}")
    print(f"  Wolfram · Math Genius · Soul Engine · 24/7")
    print(f"  Marco Barreto · Fuch F51 Labs · Montreal")
    print(f"  Soli Deo Gloria")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
