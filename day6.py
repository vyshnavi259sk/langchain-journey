from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="llama-3.1-8b-instant"
)

messages = [
    SystemMessage(content="You are a sarcastic teacher who explains things with dark humour."),
    HumanMessage(content="What is a Python virtual environment in programming?")
]

response = llm.invoke(messages)
print(response.content)