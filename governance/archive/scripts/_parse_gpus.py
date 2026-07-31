import json, os, sys

json_path = os.environ.get('VAST_JSON', sys.argv[1] if len(sys.argv) > 1 else None)
if not json_path:
    print("No JSON file")
    sys.exit(1)

data = json.load(open(json_path))

# North America filter
na_countries = ['US', 'CA', 'MX', 'United States', 'Canada', 'Mexico']
na_regions = ['California', 'Texas', 'New York', 'Florida', 'Illinois', 'Virginia', 'Georgia', 'Washington',
              'Oregon', 'Quebec', 'Ontario', 'Delaware', 'Massachusetts', 'Colorado', 'Arizona', 'Nevada',
              'New Jersey', 'North Carolina', 'Michigan', 'Ohio', 'Pennsylvania', 'Maryland', 'Minnesota',
              'Indiana', 'Tennessee', 'Wisconsin', 'Missouri', 'Connecticut', 'Utah', 'Iowa', 'Alabama',
              'South Carolina', 'Louisiana', 'Kentucky', 'Oklahoma', 'Kansas', 'Nebraska', 'Nevada',
              'British Columbia', 'Alberta', 'Montreal', 'Toronto', 'Vancouver',
              'America', 'Chicago', 'Dallas', 'Houston', 'Phoenix', 'San Jose', 'Los Angeles',
              'Seattle', 'Denver', 'Miami', 'Atlanta', 'Boston', 'Detroit', 'Portland']

# Filter: 24GB+, Turing+, North America, <= $2/h
filtered = [g for g in data if
    g.get('gpu_ram', 0) >= 24000 and
    g.get('compute_cap', 0) >= 75 and
    g.get('dph_total', 0) <= 2.0 and
    any(region.lower() in g.get('geolocation', '').lower() for region in na_regions + na_countries)
]

# Sort by reliability if available, otherwise by price
filtered.sort(key=lambda g: (
    -float(g.get('expected_reliability', 0) or 0),
    -float(g.get('reliability', 0) or 0),
    g.get('dph_total', 999)
))

print(f"=== NORTE-AMERICA: {len(filtered)} GPUs 24GB+ Turing+ <=$2/h ===")
print(f"Ordenado por: Reputacao > Preco")
print(f"Credito: $18.82")
print()
print(f"{'GPU':<22s} {'VRAM':>6s}  {'$/h':>8s}  {'3h':>7s}  {'Reliability':>12s}  {'Duration':>8s}  {'CPU':>6s}  {'Disk':>6s}  {'Location':<28s}  {'ID'}")
print('-' * 145)
for g in filtered[:20]:
    vram = f"{g['gpu_ram']/1024:.0f}GB"
    p = g['dph_total']
    cost3 = f"${p*3:.2f}"
    cpu = f"{g['cpu_ram']/1024:.0f}GB"
    disk = f"{g['disk_space']:.0f}GB"
    dur_days = g.get('duration', 0) / 86400
    dur = f"{dur_days:.0f}d"

    # Reliability: try multiple fields
    rel_raw = g.get('expected_reliability') or g.get('reliability') or g.get('reliability2') or 0
    try:
        rel = float(rel_raw)
        rel_str = f"{rel:.4f}" if rel > 0 else "N/A"
    except:
        rel_str = "N/A"

    # Downtime
    downtime = g.get('downtime_seconds', 0) or 0
    dt_str = f"{downtime/3600:.1f}h" if downtime > 0 else "N/A"

    # Verification
    verified = "V" if g.get('verified') else "-"

    print(f"{g['gpu_name']:<22s} {vram:>6s}  ${p:>6.3f}/h  {cost3:>7s}  {rel_str:>12s}  {dur:>8s}  {cpu:>6s}  {disk:>6s}  {g['geolocation']:<28s}  {g['id']}")
