import urllib.request, json
url = "https://api.github.com/users/marcobarreto007/repos?per_page=20&sort=updated"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as r:
    repos = json.loads(r.read())
print(f"PROJETOS DE MARCO BARRETO — github.com/marcobarreto007")
print(f"Total: {len(repos)} repositorios")
print()
for i, repo in enumerate(repos):
    name = repo["name"]
    lang = repo.get("language") or "?"
    desc = (repo.get("description") or "sem descricao")[:120]
    stars = repo.get("stargazers_count", 0)
    updated = repo.get("updated_at", "?")[:10]
    size = repo.get("size", 0)
    print(f"{i+1}. {name}")
    print(f"   Linguagem: {lang}  |  Stars: {stars}  |  Size: {size}KB  |  Updated: {updated}")
    print(f"   {desc}")
    print()
