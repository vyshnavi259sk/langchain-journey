import json

raw = '''
{
  "name": "Alex",
  "age": 28,
  "skills": ["Python", "SQL", "Docker"],
  "experience": {
    "current_job": "Data Analyst",
    "years": 3,
    "companies": ["Google", "Stripe"]
  },
  "open_to_work": true
}
'''


data = json.loads(raw)
print (data["name"])
print (data["skills"][1])
print (data["experience"]["years"])
print (data["experience"]["companies"][0])
print (data["open_to_work"])
data["skills"].append("LangChain")
print (data["skills"])
