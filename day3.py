import requests

response = requests.get("https://api.github.com/users/torvalds")

print(response.status_code)
data = response.json()
print(data["name"])
print(data["public_repos"])
print(data["location"])