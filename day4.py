import requests
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 is 4."},
            {"role": "user", "content": "What did I just ask you?"}
        ],
        "temperature": 0  # 0 = robotic, 1 = creative
    }
)

data = response.json()  # ← this line must be here
print(data["choices"][0]["message"]["content"])  # ← then this