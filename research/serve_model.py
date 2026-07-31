#!/usr/bin/env python
"""F51 Model Server — Local inference API for the F51 Darwin-SSD model.

Usage:
    python research/serve_model.py
    python research/serve_model.py --port 8051 --checkpoint checkpoints/base/real_train_1/step_0002000.pt

Endpoints:
    POST /generate     → Generate text from prompt
    GET  /health       → Server health + model info
    GET  /             → Web UI
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
from torch import nn

from f51_darwin.artifacts import resolve_latest_checkpoint

HTML_UI = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F51 Darwin-SSD — Console</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0f; color: #c0c0c0;
    font-family: 'Courier New', monospace;
    height: 100vh; display: flex; flex-direction: column;
}
header {
    background: #0d0d1a; padding: 12px 20px;
    border-bottom: 1px solid #1a1a2e;
    display: flex; justify-content: space-between; align-items: center;
}
.logo { color: #00ff88; font-size: 14px; font-weight: bold; }
.status { font-size: 12px; }
.status.online { color: #00ff88; }
.status.offline { color: #ff4444; }
.model-info { font-size: 11px; color: #666; margin-top: 4px; }
.container { flex: 1; display: flex; flex-direction: column; padding: 20px; max-width: 900px; margin: 0 auto; width: 100%; }
.output { flex: 1; overflow-y: auto; padding: 10px; background: #06060d; border: 1px solid #1a1a2e; border-radius: 4px; margin-bottom: 12px; }
.output .user { color: #00ccff; margin: 8px 0 4px; }
.output .user::before { content: '▸ '; }
.output .assistant { color: #c0c0c0; margin: 4px 0 8px 20px; white-space: pre-wrap; word-break: break-word; }
.output .system { color: #666; font-size: 11px; margin: 4px 0; font-style: italic; }
.input-row { display: flex; gap: 8px; }
.input-row input {
    flex: 1; padding: 10px 14px; background: #0d0d1a;
    border: 1px solid #1a1a2e; color: #c0c0c0;
    font-family: 'Courier New', monospace; font-size: 14px;
    border-radius: 4px; outline: none;
}
.input-row input:focus { border-color: #00ff88; }
.input-row button {
    padding: 10px 20px; background: #00ff88; color: #0a0a0f;
    border: none; font-weight: bold; cursor: pointer;
    font-family: 'Courier New', monospace; font-size: 14px;
    border-radius: 4px;
}
.input-row button:hover { background: #00cc66; }
.input-row button:disabled { background: #333; color: #666; cursor: not-allowed; }
.params { display: flex; gap: 12px; margin-bottom: 10px; font-size: 11px; }
.params label { color: #666; }
.params input { width: 55px; background: #0d0d1a; border: 1px solid #1a1a2e; color: #c0c0c0; padding: 3px 6px; border-radius: 3px; text-align: center; font-family: 'Courier New', monospace; }
</style>
</head>
<body>
<header>
    <div>
        <div class="logo">F51 DARWIN-SSD</div>
        <div class="model-info" id="modelInfo">carregando...</div>
    </div>
    <div class="status" id="status">conectando...</div>
</header>
<div class="container">
    <div class="output" id="output">
        <div class="system">F51 Darwin-SSD Console — Modelo local, do zero, sem checkpoint externo.</div>
        <div class="system">Criado por Marco Barreto, Fuch F51 Labs, Montréal.</div>
        <div class="system">Soli Deo Gloria.</div>
    </div>
    <div class="params">
        <label>temp <input id="temp" value="0.8" step="0.1" min="0" max="2"></label>
        <label>max tokens <input id="maxTokens" value="150" step="10" min="10" max="500"></label>
        <label>top_p <input id="topP" value="0.95" step="0.05" min="0" max="1"></label>
    </div>
    <div class="input-row">
        <input id="prompt" placeholder="Digite aqui..." autofocus onkeydown="if(event.key==='Enter')send()">
        <button id="sendBtn" onclick="send()">ENVIAR</button>
    </div>
</div>
<script>
let busy = false;

async function checkHealth() {
    try {
        let r = await fetch('/health');
        let d = await r.json();
        document.getElementById('status').textContent = 'online';
        document.getElementById('status').className = 'status online';
        document.getElementById('modelInfo').textContent =
            d.model + ' | ' + d.params_m + 'M params | ' + d.device;
    } catch(e) {
        document.getElementById('status').textContent = 'offline';
        document.getElementById('status').className = 'status offline';
    }
}

async function send() {
    if (busy) return;
    let prompt = document.getElementById('prompt').value.trim();
    if (!prompt) return;

    busy = true;
    document.getElementById('sendBtn').disabled = true;

    let out = document.getElementById('output');
    out.innerHTML += '<div class=\"user\">' + prompt + '</div>';
    out.innerHTML += '<div class=\"assistant\" id=\"loading\">gerando...</div>';
    out.scrollTop = out.scrollHeight;

    try {
        let r = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: parseInt(document.getElementById('maxTokens').value) || 150,
                temperature: parseFloat(document.getElementById('temp').value) || 0.8,
                top_p: parseFloat(document.getElementById('topP').value) || 0.95,
            })
        });
        let d = await r.json();
        document.getElementById('loading').textContent = d.text || d.error || '(erro)';
        document.getElementById('loading').id = '';
        document.getElementById('prompt').value = '';
    } catch(e) {
        document.getElementById('loading').textContent = 'Erro: ' + e.message;
        document.getElementById('loading').id = '';
    }

    busy = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('prompt').focus();
    out.scrollTop = out.scrollHeight;
}

checkHealth();
setInterval(checkHealth, 10000);
document.getElementById('prompt').focus();
</script>
</body>
</html>"""


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        raise SystemExit(1)

    class GenerateRequest(BaseModel):
        prompt: str = ""
        max_tokens: int = 150
        temperature: float = 0.8
        top_p: float = 0.95

    app = FastAPI(title="F51 Darwin-SSD", version="0.1.0")
    model = None
    tokenizer = None
    config = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(checkpoint_path: str | None = None):
        nonlocal model, tokenizer, config
        from f51_darwin.checkpointing import load_model_from_checkpoint
        from f51_darwin.tokenizer import F51BPETokenizer

        # Find latest checkpoint
        if checkpoint_path is None:
            checkpoint = resolve_latest_checkpoint(
                ROOT,
                pointers=[
                    "checkpoints/base/latest.json",
                    "checkpoints/organism/organism_latest.json",
                    "checkpoints/cloud/cloud_latest.json",
                ],
                search_roots=[
                    "checkpoints/base",
                    "checkpoints/organism",
                    "checkpoints/cloud",
                ],
            )
            checkpoint_path = None if checkpoint is None else str(checkpoint)

        if checkpoint_path and Path(checkpoint_path).exists():
            model, config, _ = load_model_from_checkpoint(checkpoint_path, map_location=DEVICE)
        else:
            from f51_darwin.config import DarwinConfig
            from f51_darwin.model import F51DarwinModel
            config = DarwinConfig()
            model = F51DarwinModel(config).to(DEVICE)

        tokenizer_dir = ROOT / "tokenizer" / "f51_bpe"
        if tokenizer_dir.exists():
            tokenizer = F51BPETokenizer.load(tokenizer_dir)

        model.eval()

    @app.on_event("startup")
    async def startup():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--port", type=int, default=8051)
        args, _ = parser.parse_known_args()
        load_model(args.checkpoint)
        print(f"F51 Server ready on port {args.port}")

    @app.get("/", response_class=HTMLResponse)
    async def ui():
        return HTML_UI

    @app.get("/health")
    async def health():
        if model is None:
            return JSONResponse({"status": "no model loaded"}, status_code=503)
        params = sum(p.numel() for p in model.parameters())
        return {
            "status": "ok",
            "model": config.model_name if config else "F51-Darwin-SSD",
            "params": params,
            "params_m": round(params / 1e6, 1),
            "device": str(next(model.parameters()).device),
            "tokenizer": tokenizer.vocab_size if tokenizer else 0,
        }

    @app.post("/generate")
    async def generate(body: GenerateRequest):
        if model is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        prompt = body.prompt
        max_tokens = min(body.max_tokens, 1024)
        temperature = body.temperature
        top_p = body.top_p

        if not prompt.strip():
            return JSONResponse({"error": "Empty prompt"}, status_code=400)

        try:
            output = model.generate(
                prompt,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return {
                "text": output.text,
                "tokens": len(output.token_ids),
                "finish_reason": output.finish_reason,
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="F51 Darwin-SSD Inference Server")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (auto-detect if omitted)")
    parser.add_argument("--port", type=int, default=8051, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn fastapi")
        raise SystemExit(1)

    # Store args for startup event
    import sys
    sys.argv = [sys.argv[0], "--port", str(args.port)]
    if args.checkpoint:
        sys.argv.extend(["--checkpoint", args.checkpoint])

    app = create_app()
    print(f"\n{'='*60}")
    print(f"  F51 Darwin-SSD Inference Server")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Model: auto-detecting latest checkpoint...")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
