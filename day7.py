from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Define a reusable template with variables in {curly braces}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at explaining {subject} concepts simply."),
    ("user", "Explain {topic} in 2 sentences.")
])

# Fill in the variables
chain = prompt | llm

response = chain.invoke({
    "subject": "cooking",
    "topic": "bittermelon"
})

print(response.content)