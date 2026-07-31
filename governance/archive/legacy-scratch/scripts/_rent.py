import urllib.request, json, time

key = "REVOKED"
bid = "39825674"

# Try renting
for attempt in range(10):
    body = json.dumps({"ask_contract_id": bid, "qty": 1}).encode()
    req = urllib.request.Request("https://console.vast.ai/api/v0/bids/", method="POST", data=body)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=8)
        result = json.loads(resp.read())
        print(f"RENTED! Instance: {result.get('instance_id','?')}")
        time.sleep(5)
        # Get SSH
        req2 = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
        req2.add_header("Authorization", f"Bearer {key}")
        insts = json.loads(urllib.request.urlopen(req2, timeout=10).read()).get("instances", [])
        for i in insts:
            print(f"GPU: {i.get('gpu_name')} {int(i.get('gpu_ram',0))//1024}GB")
            print(f"SSH: ssh -p {i.get('ssh_port')} root@{i.get('ssh_host')}")
            print(f"Disk: {i.get('disk_space')}GB Status: {i.get('actual_status')}")
        break
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:100]
        if "Not found" in err:
            print(f"Attempt {attempt+1}: expired, retrying...")
            time.sleep(2)
            continue
        print(f"Error: {err}")
        break
else:
    print("Failed after 10 attempts")
