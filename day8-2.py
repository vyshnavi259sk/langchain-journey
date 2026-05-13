"""
In this example, we chain together two prompts and two LLM calls. The output of the first 
chain becomes the input of the second chain.
The first chain generates a short blog post about a given topic, and the second chain 
summarises.
Chain -> Prompt1 | Prompt2 | ... | LLM | Parser
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Step 1 — write a short blog post about a topic
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a blog writer."),
    ("user", "Write a short 3 sentence blog post about {topic}.")
])

# Step 2 — summarise that blog post into one sentence
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are an editor who writes punchy one-line summaries."),
    ("user", "Summarise this in one sentence: {blog_post}")
])

parser = StrOutputParser()

# Chain them together
# output of chain1 becomes input of chain2
chain1 = prompt1 | llm | parser
chain2 = (
    {"blog_post": chain1}  # feed chain1's output as input to prompt2
    | prompt2
    | llm
    | parser
)

response = chain2.invoke({"topic": "why learning to code changes your life"})
print(response)