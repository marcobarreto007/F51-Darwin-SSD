import json, sys

json_path = sys.argv[1] if len(sys.argv) > 1 else None
if not json_path:
    print("No JSON file")
    sys.exit(1)

data = json.load(open(json_path))
na = ['US','CA','MX','California','Texas','New York','Florida','Illinois','Virginia','Georgia',
      'Washington','Oregon','Quebec','Ontario','Delaware','Massachusetts','Colorado','Arizona',
      'Nevada','New Jersey','North Carolina','Michigan','Ohio','Pennsylvania','Maryland','Minnesota',
      'Indiana','Tennessee','Wisconsin','Missouri','Connecticut','Utah','Iowa','Alabama',
      'South Carolina','Louisiana','Kentucky','Oklahoma','Kansas','Nebraska','British Columbia',
      'Alberta','Montreal','Toronto','Vancouver','America','Chicago','Dallas','Houston','Phoenix',
      'San Jose','Los Angeles','Seattle','Denver','Miami','Atlanta','Boston','Detroit','Portland']

filtered = [g for g in data if g.get('gpu_ram',0)>=24000 and g.get('compute_cap',0)>=75
            and any(r.lower() in g.get('geolocation','').lower() for r in na)]
filtered.sort(key=lambda g: -g.get('dlperf',0))

print(f"=== NORTE-AMERICA: FASTEST GPUs (por dlperf) ===")
print(f"Credit: $18.82 | H100 atual: $3.28/h")
print()
print(f"{'GPU':<22s} {'VRAM':>6s}  {'$/h':>8s}  {'dlperf':>8s}  {'Reliab':>7s}  {'Up':>5s}  {'Location':<25s}  {'ID'}")
print('-' * 125)
for g in filtered[:18]:
    dlp = g.get('dlperf',0)
    rel = g.get('reliability2',0) or 0
    p = g.get('dph_total',0)
    dur = g.get('duration',0)/86400
    # calc hours of credit
    hrs = 16.66/p if p > 0 else 999
    vram = g['gpu_ram']/1024
    marker = ""
    if dlp > 300: marker = " ⚡"
    if dlp > 500: marker = " 🔥"
    if rel > 0.999: marker += " 💎"
    print(f"{g['gpu_name']:<22s} {vram:>5.0f}GB  ${p:>6.3f}/h  {dlp:>7.0f}{marker:>3s}  {rel:>6.4f}  {dur:>5.0f}d  {g['geolocation']:<25s}  {g['id']}  ({hrs:.0f}h cred)")

# Show current instances for comparison
print(f"\n=== COMPARACAO COM ATUAIS ===")
print(f"{'A6000 atual':<22s} {'45GB':>6s}  ${0.403:>6.3f}/h  {51:>7.0f}   {0.998:>6.4f}  {'183d':>5s}  {'Delaware, US':<25s}  45344965")
print(f"{'H100 atual':<22s} {'80GB':>6s}  ${3.278:>6.3f}/h  {352:>7.0f}   {0.998:>6.4f}  {'26m':>5s}  {', US':<25s}  45349389 <- SEM GPU UTIL!")
