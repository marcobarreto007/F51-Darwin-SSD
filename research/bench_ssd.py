import torch, time
from f51_darwin.ssm_core import selective_scan_parallel
device = torch.device("cuda")
dim, seq, inner = 1408, 512, 1408*2
u = torch.randn(1, inner, seq, device=device)
delta = torch.randn(1, inner, seq, device=device)
a = -torch.exp(torch.randn(inner, 16, device=device))
b = torch.randn(1, 16, seq, device=device)
c = torch.randn(1, 16, seq, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): y1 = selective_scan_parallel(u, delta, a, b, c)
torch.cuda.synchronize()
tp = time.time() - t0
compiled = torch.compile(selective_scan_parallel, backend="inductor", fullgraph=True)
_ = compiled(u, delta, a, b, c)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): y2 = compiled(u, delta, a, b, c)
torch.cuda.synchronize()
tt = time.time() - t0
print(f"Python: {tp:.3f}s | Triton: {tt:.3f}s | SPEEDUP: {tp/tt:.1f}x")
