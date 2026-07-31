import urllib.request, json
url = "https://api.github.com/repos/marcobarreto007/F51-Darwin-SSD"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read())
    print(f"F51-Darwin-SSD EXISTE no GitHub")
    print(f"  Private: {d.get('private')}")
    print(f"  Size: {d.get('size', 0)} KB")
    print(f"  Language: {d.get('language')}")
    print(f"  Description: {d.get('description', 'sem')}")
    print(f"  Updated: {d.get('updated_at', '?')[:10]}")
    print(f"  Default branch: {d.get('default_branch')}")
except Exception as e:
    print(f"F51-Darwin-SSD: {e}")
