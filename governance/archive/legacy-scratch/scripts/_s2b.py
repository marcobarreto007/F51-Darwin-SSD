import urllib.request, json
key = "REVOKED"
req = urllib.request.Request("https://console.vast.ai/api/v0/instances/", method="GET")
req.add_header("Authorization", f"Bearer {key}")
insts = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("instances", [])
for i in insts:
    print(f"Direct: {i.get('public_ipaddr')}:{i.get('direct_port_start')}")
    print(f"Proxy: {i.get('ssh_host')}:{i.get('ssh_port')}")
    print(f"Inet up: {i.get('inet_up')} Mbps")
