import urllib.request, json
key = "REVOKED"
req = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
req.add_header("Authorization", f"Bearer {key}")
insts = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("instances", [])
print(f"Active: {len(insts)}")
for i in insts:
    name = i.get("gpu_name", "?")
    ram = int(i.get("gpu_ram", 0)) // 1024
    dph = float(i.get("dph_total", 0))
    disk = float(i.get("disk_space", 0))
    ssh_host = i.get("ssh_host", "?")
    ssh_port = i.get("ssh_port", "?")
    status = i.get("actual_status", "?")
    credit = 27.17
    hours = credit / dph if dph > 0 else 0
    print(f"  {name} {ram}GB disk={disk:.0f}GB usd={dph:.2f}/h ({hours:.0f}h) {status}")
    print(f"  SSH: ssh -p {ssh_port} root@{ssh_host}")
