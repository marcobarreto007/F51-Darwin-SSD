import urllib.request, json
url = "https://api.github.com/users/marcobarreto007/repos?sort=updated&per_page=10"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as r:
    repos = json.loads(r.read())
print("All repos:")
for repo in repos:
    desc = (repo.get('description') or 'no description')[:80]
    stars = repo.get('stargazers_count', 0)
    lang = repo.get('language', '?')
    print(f"  {repo['name']:<35} {lang:<12} stars={stars}  {desc}")
