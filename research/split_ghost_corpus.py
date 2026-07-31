#!/usr/bin/env python
"""F51 Ghost Corpus Splitter — Divide corpus em 80% treino / 20% fantasma"""
import sys, random, json, shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]

def main():
    corpus_dir = Path("data/corpus")
    ghost_dir = Path("data/ghost_corpus")
    train_dir = Path("data/train_corpus")
    
    # Clean previous
    if ghost_dir.exists():
        shutil.rmtree(ghost_dir)
    if train_dir.exists():
        shutil.rmtree(train_dir)
    
    ghost_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    
    # Collect all text files
    all_files = []
    for f in corpus_dir.rglob("*.txt"):
        if f.is_file():
            all_files.append(f)
    
    random.seed(42)
    random.shuffle(all_files)
    
    split = int(len(all_files) * 0.20)
    ghost_files = all_files[:split]
    train_files = all_files[split:]
    
    # Copy ghost files
    for f in ghost_files:
        dest = ghost_dir / f.name
        shutil.copy2(f, dest)
    
    # Copy train files
    for f in train_files:
        dest = train_dir / f.name
        shutil.copy2(f, dest)
    
    ghost_size = sum(f.stat().st_size for f in ghost_files) / 1e9
    train_size = sum(f.stat().st_size for f in train_files) / 1e9
    
    # Domain breakdown
    ghost_domains = defaultdict(lambda: {"files": 0, "size": 0})
    for f in ghost_files:
        name = f.name.lower()
        if 'math' in name: d = 'math'
        elif 'physics' in name or 'quantum' in name: d = 'physics'
        elif 'finance' in name or 'trading' in name or 'market' in name: d = 'finance'
        elif 'bio' in name: d = 'biology'
        elif 'code' in name or 'python' in name: d = 'code'
        else: d = 'general'
        ghost_domains[d]["files"] += 1
        ghost_domains[d]["size"] += f.stat().st_size
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  F51 GHOST CORPUS — 80/20 SPLIT                 ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Total files:    {len(all_files)}                              ║")
    print(f"║  Treino (80%):   {len(train_files)} files, {train_size:.1f} GB       ║")
    print(f"║  Fantasma (20%): {len(ghost_files)} files, {ghost_size:.1f} GB       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Fantasma por domínio:                           ║")
    for domain in sorted(ghost_domains):
        info = ghost_domains[domain]
        print(f"║    {domain:<12s}: {info['files']:>3d} files, {info['size']/1e9:.1f} GB              ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"📁 Treino:    {train_dir}/")
    print(f"📁 Fantasma:  {ghost_dir}/")
    
    # Save manifest
    manifest = {
        "split_date": datetime.now(timezone.utc).isoformat(),
        "total_files": len(all_files),
        "train_files": len(train_files),
        "ghost_files": len(ghost_files),
        "train_size_gb": round(train_size, 2),
        "ghost_size_gb": round(ghost_size, 2),
    }
    (ghost_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    print()
    print("Pipeline de auto-evolução:")
    print("  1. Treinar modelo no train_corpus/")
    print("  2. CuriosityDrive explora ghost_corpus/")
    print("  3. Modelo gera perguntas sobre o desconhecido")
    print("  4. Wolfram/Code verifica respostas")
    print("  5. Dados sintéticos alimentam auto-treino")
    print("  6. Modelo supera os dados originais 🧬")

if __name__ == "__main__":
    main()
