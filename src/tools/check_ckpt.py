import sys
import torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

if len(sys.argv) > 1:
    ckpt_path = Path(sys.argv[1])
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
else:
    ckpt_path = ROOT / 'checkpoints' / 'organism' / 'organism_247' / 'step_0006000.pt'

if not ckpt_path.exists():
    print(f'Checkpoint not found: {ckpt_path}', file=sys.stderr)
    print(f'Usage: python {Path(__file__).name} <relative_or_absolute_ckpt_path>', file=sys.stderr)
    sys.exit(1)

payload = torch.load(str(ckpt_path), map_location='cpu')
print(f'loaded: {ckpt_path}')
print('payload keys:', list(payload.keys()))
if 'training_state' in payload:
    print('training_state keys:', list(payload['training_state'].keys()))
    print('step:', payload['training_state'].get('step'))
