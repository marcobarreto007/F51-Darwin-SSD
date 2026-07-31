import urllib.request, json, time, concurrent.futures

key = "REVOKED"
req = urllib.request.Request("https://console.vast.ai/api/v0/bundles/", method="GET")
req.add_header("Authorization", f"Bearer {key}")
offers = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("offers", [])

# Collect ALL rentable with >= 40 GB VRAM and >= 200 GB disk
viable = []
for o in offers:
    if o.get("rentable") != True: continue
    ram = int(o.get("gpu_ram", 0))
    if ram < 40000: continue
    disk = float(o.get("disk_space", 0))
    if disk < 200: continue
    dph = float(o.get("dph_total", 0))
    if dph > 3.0: continue
    bid = str(o.get("ask_contract_id", ""))
    name = o.get("gpu_name", "?")
    viable.append((bid, name, ram//1024, disk, dph))

print(f"Viable: {len(viable)}")

def try_bid(bid, name, ram, disk, dph):
    body = json.dumps({"ask_contract_id": bid, "qty": 1}).encode()
    req = urllib.request.Request("https://console.vast.ai/api/v0/bids/", method="POST", data=body)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=8)
        return (True, bid, name, ram, disk, dph, json.loads(resp.read()))
    except:
        return (False, bid, name, ram, disk, dph, None)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(try_bid, *v): v for v in viable}
    for future in concurrent.futures.as_completed(futures):
        ok, bid, name, ram, disk, dph, result = future.result()
        if ok:
            print(f"RENTED! {name} {ram}GB disk={disk:.0f}GB usd={dph:.2f}/h")
            time.sleep(5)
            req2 = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
            req2.add_header("Authorization", f"Bearer {key}")
            insts = json.loads(urllib.request.urlopen(req2, timeout=10).read()).get("instances", [])
            for i in insts:
                print(f"SSH: ssh -p {i.get('ssh_port')} root@{i.get('ssh_host')}")
                print(f"Disk: {i.get('disk_space')}GB")
            break
        else:
            print(f"  fail: {name}")
else:
    print("All failed.")
