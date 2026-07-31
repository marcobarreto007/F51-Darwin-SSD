#!/bin/bash
# Rent GPU with reputation check
set -e

echo "🔍 Buscando GPUs com reputation check..."

# Get top 5 by reliability
OFFERS=$(vastai search offers "rentable=true verified=true num_gpus=1 gpu_ram>=24000 reliability>=0.97" --order reliability --limit 10 --raw 2>/dev/null)

# Filter: duration > 30 days, downtime < 1h, disk >= 200GB
BEST=$(echo "$OFFERS" | python3 -c "
import sys,json
data = json.load(sys.stdin)
for o in data:
    rel = o.get('reliability', 0)
    dur = o.get('duration', 0) / 86400
    down = o.get('downtime_seconds', 0) / 3600
    disk = o.get('disk_space', 0)
    gpu = o.get('gpu_name', '?')
    vram = o.get('gpu_ram', 0) / 1024
    dph = o.get('dph_total', 0)
    loc = o.get('geolocation', '?')
    if dur > 30 and down < 1 and disk >= 200:
        print(f'{o[\"id\"]}|{gpu}|{vram:.0f}GB|\${dph:.2f}|r={rel:.2f}|{dur:.0f}d|{disk:.0f}GB|{loc[:20]}')
        break
")

if [ -z "$BEST" ]; then
  echo "❌ Nenhuma GPU com reputation boa. Tentando em 10min..."
  exit 1
fi

ID=$(echo $BEST | cut -d'|' -f1)
echo "✅ Alugando: $BEST"

vastai create instance "$ID" --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 200 --ssh --direct
echo "GPU_ID=$ID" > /tmp/last_gpu.txt
