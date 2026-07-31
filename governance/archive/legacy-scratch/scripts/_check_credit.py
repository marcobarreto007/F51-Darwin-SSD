import urllib.request, json
key = "REVOKED"
req = urllib.request.Request("https://console.vast.ai/api/v0/users/current/", method="GET")
req.add_header("Authorization", f"Bearer {key}")
data = json.loads(urllib.request.urlopen(req, timeout=10).read())
credit = data.get("credit", 0)
spent = abs(data.get("total_spend", 0))
print(f"Credit: usd={credit:.2f}")
print(f"Total spent: usd={spent:.2f}")
print()

req2 = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
req2.add_header("Authorization", f"Bearer {key}")
insts = json.loads(urllib.request.urlopen(req2, timeout=10).read()).get("instances", [])
print(f"Active instances: {len(insts)}")
for i in insts:
    name = i.get("gpu_name", "?")
    ram = int(i.get("gpu_ram", 0)) // 1024
    dph = float(i.get("dph_total", 0))
    hours_left = credit / dph if dph > 0 else 0
    status = i.get("actual_status", "?")
    print(f"  {name} {ram}GB usd={dph:.2f}/h = {hours_left:.1f}h left | {status}")
if not insts:
    print("  NO ACTIVE INSTANCES")
