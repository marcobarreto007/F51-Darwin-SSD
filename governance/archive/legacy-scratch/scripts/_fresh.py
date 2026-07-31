import urllib.request, json, time

key = "REVOKED"

# Destroy current
req = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
req.add_header("Authorization", f"Bearer {key}")
insts = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("instances", [])
for i in insts:
    iid = i.get("id")
    print(f"Destroying instance {iid}...")
    req2 = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{iid}/", method="DELETE")
    req2.add_header("Authorization", f"Bearer {key}")
    resp = json.loads(urllib.request.urlopen(req2, timeout=10).read())
    print(f"Destroyed: {resp}")

# Check credit
req3 = urllib.request.Request("https://console.vast.ai/api/v0/users/current/", method="GET")
req3.add_header("Authorization", f"Bearer {key}")
data = json.loads(urllib.request.urlopen(req3, timeout=10).read())
credit = data.get("credit", 0)
print(f"Credit: usd={credit:.2f}")

# Search for A100 or RTX PRO 6000
time.sleep(2)
req4 = urllib.request.Request("https://console.vast.ai/api/v0/bundles/", method="GET")
req4.add_header("Authorization", f"Bearer {key}")
offers = json.loads(urllib.request.urlopen(req4, timeout=10).read()).get("offers", [])
print("Available GPUs:")
for o in offers:
    if o.get("rentable") != True: continue
    name = str(o.get("gpu_name", "?"))
    ram = int(o.get("gpu_ram", 0)) // 1024
    if ram < 40: continue
    dph = float(o.get("dph_total", 0))
    disk = float(o.get("disk_space", 0))
    if dph > 2.0: continue
    loc = o.get("geolocation", "?")
    hours = credit / dph if dph > 0 else 0
    print(f"  {name} {ram}GB disk={disk:.0f}GB usd={dph:.2f}/h ({hours:.0f}h) {loc} bid={o.get('ask_contract_id','')}")
