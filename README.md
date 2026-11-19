# LangChain Learning Journey

A structured, day-by-day log of learning LangChain and LLM API integration from first principles — starting with raw JSON/HTTP API calls and building up to multi-step LLM chains.

**Phase 1 — Fundamentals (Days 1–4):** JSON parsing, HTTP requests, calling an AI API directly, environment setup.
**Phase 2 — LangChain (Days 6–8):** prompt templates, chains, output parsers, and multi-step LLM pipelines using Groq + Llama 3.1.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with:

```
GROQ_API_KEY=your_key_here
```

(Groq provides the OpenAI-compatible API used throughout this repo — see [groq.com](https://groq.com).)

## Day-by-day

| Day | File | What it covers |
|-----|------|-----------------|
| 1 | `day1.py` | Parsing nested JSON, indexing into dicts/lists, mutating a parsed structure |
| 2 | `day2.py` | Parsing a chat-completion-shaped JSON response; serializing a dict back to JSON with `json.dumps` |
| 3 | `day3.py` | Making a live HTTP GET request with `requests` and reading the JSON response |
| 4 | `day4.py` | Calling an LLM chat-completion endpoint (Groq) directly over HTTP with `requests`, using an API key loaded via `dotenv` |
| 6 | `day6.py` | First LangChain exercise: `ChatGroq` + `SystemMessage`/`HumanMessage`, invoking a chat model directly |
| 7 | `day7.py` | `ChatPromptTemplate` with variables, and the `prompt \| llm` chain syntax |
| 8 | `day8-1.py` | Full chain: `prompt \| llm \| StrOutputParser`, so `.invoke()` returns plain text instead of a message object |
| 8 | `day8-2.py` | Chaining two prompts together — the output of one LLM call feeds into the next (`{"blog_post": chain1} \| prompt2 \| llm \| parser`) |

(Day 5 was a planning/reading day with no code checked in.)

`journal.md` also tracks its own "Day N" count, separate from the table
above (it started over from Day 1 rather than continuing from Day 8). Its
implementation files are named to avoid colliding with the table above:

| Journal day | File | What it covers |
|-----|------|-----------------|
| "Day 2" (2026-11-14) | `day2_lcel_runnables.py` | LCEL's `RunnableParallel`, `RunnableLambda`, and `RunnablePassthrough` — fanning a single input into multiple branches, the shape that later powers RAG chains |
| "Day 3" (2026-11-15) | `day3_document_loaders.py` | `TextLoader`, `DirectoryLoader`, `PyPDFLoader`, and `WebBaseLoader` turning files/PDFs/web pages into `Document` objects with `page_content` + `metadata` |
| "Day 4" (2026-11-15) | `day4_text_splitting.py` | `CharacterTextSplitter` and `RecursiveCharacterTextSplitter` — breaking `Document`s into overlapping chunks via `chunk_size`/`chunk_overlap` ahead of embeddings |
| "Day 5" (2026-11-16) | `day5_embeddings_vectorstore.py` | A custom `Embeddings` subclass (`TfidfEmbeddings`, local/offline, no API key) plus `langchain_community.vectorstores.FAISS` — `from_documents`, `similarity_search`, `similarity_search_with_score`, `as_retriever`, and `add_documents` |
| "Day 6" (2026-11-17) | `day6_retrieval_rag.py` | A full RAG chain — `RunnableParallel(context=retriever \| RunnableLambda(format_docs), question=RunnablePassthrough()) \| prompt \| llm \| StrOutputParser()` — plus the `k` recall/precision tradeoff |
| "Day 7" (2026-11-18) | `day7_conversation_memory.py` | Conversation memory via LangGraph's `StateGraph`/`MessagesState` + `InMemorySaver` checkpointer (the current, non-deprecated replacement for `RunnableWithMessageHistory`), keyed by `thread_id`, plus `trim_messages` to cap history sent to the model |
| "Day 8" (2026-11-19) | `day8_conversational_rag.py` | Conversational RAG — a history-aware query-rewriting step (`contextualize_question()`) in front of the Day 6 retriever, combined with Day 7's `StateGraph` + checkpointer, so follow-up questions that rely on prior turns still retrieve the right context |

## Daily log

`journal.md` tracks a running streak of daily commits to this repo.
