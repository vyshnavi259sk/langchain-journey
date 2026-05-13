import json

raw_response = '''
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 45,
    "total_tokens": 65
  },
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "LangChain is a framework for building LLM-powered applications."
      },
      "finish_reason": "stop"
    }
  ]
}
'''


data = json.loads(raw_response)
print(data["model"])
print(data["choices"][0]["message"]["content"])
print(data["usage"]["total_tokens"])
print(data["choices"][0]["finish_reason"])

data= {
  "model_used": "gpt-4",
  "reply": "LangChain is a framework...",
  "tokens_spent": 65
}

raw = json.dumps(data, indent=2)
print(raw)