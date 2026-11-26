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
| "Day 9" (2026-11-20) | `day9_tool_calling.py` | Tool calling — `@tool`-decorated functions, `ChatGroq.bind_tools()`, and a hand-rolled call → execute → `ToolMessage` feed-back loop (the mechanics an agent framework later automates) |
| "Day 10" (2026-11-21) | `day10_react_agent.py` | Agents — the Day 9 tool loop rebuilt as an explicit LangGraph agent/tools graph via `create_react_agent`, with `ToolNode` and conditional edges replacing the hand-written `while` loop |
| "Day 11" (2026-11-22) | `day11_agentic_rag.py` | Agentic RAG — the Day 5/6/8 FAISS retriever wrapped as a plain `@tool` alongside an unrelated tool, so a `create_react_agent` decides per-question whether retrieval is even relevant instead of a fixed pipeline always retrieving first |
| "Day 12" (2026-11-23) | `day12_multi_agent_supervisor.py` | Multi-agent systems — a LangGraph supervisor node classifies each question into `research`/`math`/`direct` and routes it to a dedicated `create_react_agent` specialist (research-only or math-only tools) instead of one agent holding every tool |
| "Day 13" (2026-11-24) | `day13_agent_handoffs.py` | Agent handoffs — `langgraph.types.Command` lets a specialist return a state update *and* a routing decision in one value, so a compound question can hop directly from one specialist to the other (with a `visited` cycle guard) instead of always returning through a central supervisor |
| "Day 14" (2026-11-25) | `day14_human_in_the_loop.py` | Human-in-the-loop agents — `langgraph.types.interrupt()` pauses a graph before a sensitive tool call (e.g. `send_notification`), surfacing the pending call for approval; `Command(resume=...)` resumes into the real tool call or a rejection `ToolMessage`, while safe tools (`add_numbers`) still run immediately with no pause |
| "Day 15" (2026-11-26) | `day15_persistent_checkpointing.py` | Persistent checkpointing — swapping Day 7/14's in-memory `MemorySaver` for `langgraph.checkpoint.sqlite.SqliteSaver`, proving a paused `interrupt()` thread survives a simulated process restart (a brand-new SQLite connection resumes it from disk) and that separate `thread_id`s on the same file never cross-contaminate, plus `get_state_history()` for checkpoint-by-checkpoint time travel |

## Daily log

`journal.md` tracks a running streak of daily commits to this repo.
