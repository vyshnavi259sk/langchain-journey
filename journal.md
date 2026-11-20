# Daily Journal

## 2026-11-15
Day 1 of daily commits. Continuing to build consistency.

## 2026-11-14
**Day 2: LCEL Runnables — RunnableParallel, RunnableLambda, and RunnablePassthrough**

Day 8 already chained two prompts with the `|` operator, which is LCEL's `RunnableSequence` under the hood. Today's focus is the rest of the Runnable interface that makes LCEL composable beyond a straight line:

- **RunnablePassthrough** — forwards an input (or a sub-key of it) unchanged into a later step, so the original user question can still reach the final prompt even after an intermediate step (like a retriever) has transformed the data.
- **RunnableParallel** — runs multiple branches on the same input at once and merges their outputs into a dict, e.g. `RunnableParallel(context=retriever, question=RunnablePassthrough())`. This is the pattern that later powers RAG chains, where "context" and "question" need to reach the prompt together.
- **RunnableLambda** — wraps a plain Python function as a Runnable so custom logic (formatting, filtering, light post-processing) can sit inside a `|` chain without breaking the interface.

Practical exercise: build a chain that takes a topic string, fans it out with `RunnableParallel` into two branches — one that passes the topic through unchanged and one that wraps a small Python function (`RunnableLambda`) to uppercase it — then merges both into a prompt template before sending to the LLM. This mirrors the shape of a real retrieval chain (`{"context": retriever, "question": RunnablePassthrough()}`) without needing a vector store yet, which is the next topic in the roadmap (document loaders → embeddings → vector stores → retrieval).

## 2026-11-15
**Day 3: Document Loaders — TextLoader, PyPDFLoader, DirectoryLoader, and WebBaseLoader**

With the Runnable interface in hand, the next step in the roadmap is getting real-world data into a chain at all — LangChain's **document loaders**. Every loader returns a list of `Document` objects, each holding `page_content` (the raw text) and `metadata` (a dict with things like `source`, `page`, or any field the loader attaches), so downstream steps (splitting, embedding, retrieval) can rely on one consistent shape no matter where the data came from.

Key loaders covered today:

- **TextLoader** — reads a single plain-text file into one `Document`. The simplest case, useful for confirming the `Document` shape before moving to messier sources.
- **PyPDFLoader** — parses a PDF and returns one `Document` per page, with `metadata["page"]` set automatically. This page-level split matters later: it's a natural, pre-existing chunk boundary that text splitting can respect.
- **DirectoryLoader** — walks a folder and applies a loader (e.g. `TextLoader`) to every matching file via a glob pattern, batching many files into a single list of `Document`s without hand-written loops.
- **WebBaseLoader** — fetches a URL with `requests` and strips HTML with `BeautifulSoup` to produce a `Document`, with `metadata["source"]` set to the page URL. This is the bridge from "my own files" to "the live web" as a data source.

Practical exercise: load three different sources — a local `.txt` file with `TextLoader`, a small PDF with `PyPDFLoader`, and a folder of notes with `DirectoryLoader` — then print `len(documents)` and `documents[0].metadata` for each to compare how metadata differs by loader. This sets up the next topic directly: these raw `Document`s are usually too large to embed as-is, so **text splitting** (e.g. `RecursiveCharacterTextSplitter`) is the next stop before embeddings and vector stores.

## 15-11-2026
**Day 4: Text Splitting — CharacterTextSplitter and RecursiveCharacterTextSplitter**

Day 3's loaders produce `Document` objects, but those Documents are usually far too large to embed or feed into an LLM's context window as-is. Today's topic is **text splitting**: breaking a Document's `page_content` into smaller, slightly overlapping chunks before the next steps (embeddings → vector store → retrieval).

Two key parameters control every splitter:

- **`chunk_size`** — the target maximum size of each chunk (characters, for the splitters used here).
- **`chunk_overlap`** — how many characters repeat between the end of one chunk and the start of the next, so a sentence sitting on a chunk boundary still appears whole in at least one chunk and local context isn't lost.

Splitters covered, both from the standalone `langchain_text_splitters` package (the current recommended import; `langchain.text_splitter` still re-exports them for backwards compatibility):

- **`CharacterTextSplitter`** — splits on a single separator (default `"\n\n"`), only falling back to a hard character cut if a resulting piece is still bigger than `chunk_size`. Simple, but can produce chunks well over the target size if paragraphs are long, and prints a warning when that happens.
- **`RecursiveCharacterTextSplitter`** — tries a *list* of separators in order (paragraph → line → sentence-ish → word → character), recursing into whichever piece is still too big. This is LangChain's recommended default for generic text because it keeps semantically related text together far more reliably than a single fixed separator.

Both expose `split_text(str) -> list[str]` for raw strings and `split_documents(list[Document]) -> list[Document]`, which splits `page_content` while copying the parent's `metadata` onto every resulting chunk — so each chunk still knows which file (and page, for PDFs) it came from, which retrieval will depend on later.

Practical implementation: `day4_text_splitting.py` runs both splitters on the same deliberately "messy" sample text (short title line, one long paragraph, a second paragraph, and a closing line) with identical `chunk_size`/`chunk_overlap` settings and prints each splitter's chunks side by side — `CharacterTextSplitter` produces 4 chunks (two of them over the 220-character target, with a visible warning), while `RecursiveCharacterTextSplitter` produces 7 chunks, all within the target size, by cutting on sentence/word boundaries instead. It then runs `split_documents()` on two sample `Document`s and asserts that `metadata["source"]` survives onto every chunk. The exercise at the end re-runs the comparison with a much smaller `chunk_size=60`, showing both splitters produce far more chunks, with `RecursiveCharacterTextSplitter` still favoring word boundaries where `CharacterTextSplitter` cannot.

No API key or network access is required — this is pure, offline Python. Next up: turning these chunks into embeddings and storing them in a vector database.

## 16-11-2026
**Day 5: Embeddings and Vector Stores**

Day 4 left off with a pile of small text chunks, each still carrying `metadata["source"]`. Those chunks are only useful for *semantic* search once each one is converted into an embedding — a fixed-length vector of floats that places similar meanings close together in vector space — and stored somewhere that can search by similarity instead of exact keyword matching. That's today's pair of concepts: **embeddings** and **vector stores**.

Key pieces:

- **`langchain_core.embeddings.Embeddings`** — the abstract base class every embedding backend (OpenAIEmbeddings, HuggingFaceEmbeddings, etc.) implements. It only needs two methods: `embed_documents(texts) -> list[list[float]]` and `embed_query(text) -> list[float]`. Because every backend shares this interface, a vector store built against one embedding model can be swapped to another without touching any other code.
- **`langchain_community.vectorstores.FAISS`** — LangChain's wrapper around Facebook AI Similarity Search, a local, in-memory vector index with no server and no API key. It exposes `.from_documents()`, `.similarity_search()`, `.similarity_search_with_score()`, `.add_documents()`, and `.as_retriever()`.
- **Why it matters**: this is the index a Retrieval-Augmented Generation (RAG) chain queries at run time instead of relying only on what an LLM memorized during training — and `.as_retriever()` returns a standard LangChain `Runnable`, so it plugs directly into the `RunnableParallel` pattern from Day 2 (`{"context": retriever, "question": RunnablePassthrough()}`) to build a real RAG chain next.

Groq (used for chat in day6.py/day7.py/day8-*.py) doesn't serve an embeddings endpoint, and a real sentence-transformer model would require a large torch download. So today's practical implementation, `day5_embeddings_vectorstore.py`, writes a small custom `Embeddings` subclass, `TfidfEmbeddings`, backed by scikit-learn's `TfidfVectorizer` — a legitimate free/local embedding method with no API key and no network call. It builds a FAISS index from six documents about this same learning journey (day3/day4/day2/day7/day8-1 concepts), runs `similarity_search` and `similarity_search_with_score` against a couple of natural-language queries and confirms the nearest matches are the semantically correct ones, demonstrates `as_retriever().invoke(...)`, and finishes with an exercise that calls `add_documents()` to grow the index after it's already built and re-queries to confirm the new document is retrievable. A commented-out block at the bottom shows the one-line swap to a real API-based `OpenAIEmbeddings` (reading its key via `os.getenv("OPENAI_API_KEY")`) with everything downstream unchanged. `requirements.txt` gained `numpy`, `scikit-learn`, and `faiss-cpu`; no new environment variables were needed since the script runs fully offline.

## 17-11-2026
**Day 6: Retrieval-Augmented Generation (RAG) — wiring a retriever into an LCEL chain**

Day 5 ended with a FAISS vector store whose `.as_retriever()` is just another LangChain `Runnable`, and Day 2 covered `RunnableParallel`/`RunnablePassthrough`, the pieces that fan one input into several branches and merge the results into a dict. Today puts both together to build an actual **Retrieval-Augmented Generation (RAG) chain** — the pattern behind every "chat with your own documents" system.

The core LCEL expression:

```python
rag_chain = (
    RunnableParallel(context=retriever | RunnableLambda(format_docs), question=RunnablePassthrough())
    | prompt
    | llm
    | StrOutputParser()
)
```

Key components and how they fit:

- **`RunnableParallel(context=..., question=...)`** — takes the raw question string and runs two branches on it at once: the retriever turns it into relevant `Document`s ("context"), while `RunnablePassthrough()` forwards the original question unchanged. The results merge into `{"context": ..., "question": ...}`.
- **`RunnableLambda(format_docs)`** — a small custom function wrapped as a Runnable that joins the retrieved `Document` list into a single string, since a prompt template needs text, not Python objects.
- **`ChatPromptTemplate`** — a system message instructing the model to answer *only* from the supplied context (so it doesn't fall back on parametric knowledge), plus the merged dict filling in the human message.
- **`ChatGroq` + `StrOutputParser()`** — generates the grounded answer and unwraps it to plain text, same as day8-1.py.

Why this matters: an LLM's training data is frozen and has no knowledge of documents that only live in a private vector store. RAG fixes that at query time by retrieving the most relevant chunks and stuffing them into the prompt before generation, trading a cheap retrieval step for much better grounding and far less hallucination on domain-specific questions.

Practical implementation: `day6_retrieval_rag.py` rebuilds a Day-5-style `TfidfEmbeddings` + FAISS index over a small knowledge base of facts about this very project (deliberately specific enough that a general-purpose LLM couldn't already know them), then assembles the full `RunnableParallel → prompt → llm → StrOutputParser` chain described above. Since this environment has no `GROQ_API_KEY` configured, the script detects that at chain-build time, skips constructing `ChatGroq` entirely, and instead invokes just the retrieval + prompt-assembly stages — printing the retrieved context and the exact rendered prompt messages that would have been sent — so the LCEL wiring is demonstrated end-to-end without a network call (it still prefers a real answer via Groq when a key is present). The exercise at the end compares `k=1` vs `k=3` on the same query to show how the retriever's `k` controls how much (and how precise) the retrieved context is — the classic recall/precision tradeoff in retrieval. No new dependencies or environment variables were needed; `.env.example` was updated to note that this script also runs correctly without `GROQ_API_KEY` set.

## 18-11-2026
**Day 7: Conversation Memory and Chat History — LangGraph's `StateGraph` + checkpointer**

Day 6's RAG chain answers one question well but has no memory: a follow-up like "what about that?" means nothing to it, because every `.invoke()` is independent. Today's topic is **conversation memory** — carrying prior turns forward so follow-ups work.

The first draft of this used the "classic" LangChain pattern still shown in a lot of tutorials — `RunnableWithMessageHistory` wrapping a chain, backed by `InMemoryChatMessageHistory` — but running it against the version of `langchain-core` installed in this project immediately raised `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.` So the implementation was rebuilt on the currently-recommended replacement instead:

- **`langgraph.graph.StateGraph` + `MessagesState`** — a tiny graph (here, one node) whose shared state has a `messages` list that *merges* (appends) on every update rather than overwriting, via LangGraph's built-in `add_messages` reducer. A node just returns `{"messages": [new_message]}` and the graph takes care of appending it to the running history.
- **`langgraph.checkpoint.memory.InMemorySaver`** — a checkpointer that saves the graph's state after each step, keyed by a `thread_id` passed in `config={"configurable": {"thread_id": ...}}`. Re-using a `thread_id` on a later `.invoke()` resumes that exact conversation; a new `thread_id` starts a fully isolated one. This is the direct replacement for the deprecated API's `session_id` concept.
- **`trim_messages`** (`langchain_core.messages`, still current/non-deprecated) — caps how many messages actually reach the model on each turn, independent of how much the checkpointer has stored, since history grows forever but a context window (and a token bill) doesn't.

Why it matters: multi-turn memory is table stakes for almost any real LLM product, and LangGraph's checkpointer is also exactly the mechanism the roadmap's later agent/multi-agent topics (Day 31+) are built on — so this doubles as a first, deliberately small look at LangGraph before using it for full agents.

Practical implementation: `day7_conversation_memory.py` compiles a one-node LangGraph app (`call_model` trims history, prepends a system prompt, and gets a reply) with an `InMemorySaver` checkpointer, then demonstrates memory across two turns on one `thread_id` (the second question, "what is my favorite programming language?", only makes sense given the first turn's answer) and proves a second `thread_id` has zero knowledge of the first by calling `app.get_state()` directly against the checkpointer. Since this environment has no `GROQ_API_KEY`, the "respond" step falls back to a deterministic, free, local stand-in (with basic error handling: a live Groq call that raises any exception also falls back to it) rather than skipping the demo, so the full graph/state/checkpointer plumbing still runs end-to-end with no network call — it prefers a real Groq reply whenever a key is present. The exercise re-runs `trim_messages` over a longer 6-message conversation at three different `max_tokens` budgets to show progressively more of the older history getting dropped before the model ever sees it. New dependency: `langgraph` (added to `requirements.txt`); no new environment variables were needed, and `.env.example` / `README.md` were updated to mention the new file.

## 19-11-2026
**Day 8: Conversational RAG — history-aware retrieval with LangGraph**

Day 6 (`day6_retrieval_rag.py`) built a RAG chain that answers one question well, but treats every `.invoke()` as independent — a follow-up that leans on a pronoun ("what about that?") gets embedded and searched exactly as written, with no idea what "that" refers to. Day 7 (`day7_conversation_memory.py`) separately solved multi-turn memory with a LangGraph `StateGraph` + `InMemorySaver` checkpointer, but that graph never retrieved anything. Today combines both into a single **conversational RAG** graph — the "history-aware retriever" pattern from LangChain's own tutorials, rebuilt on LangGraph instead of the deprecated `RunnableWithMessageHistory`.

What the concept is: on every turn, the graph node (1) takes the full message history from the checkpointer, (2) uses a second, small LLM call to rewrite the latest, possibly context-dependent question into a standalone one ("what about that?" → "What does this project's FAISS retriever use to build its vectors? What about that?"), (3) retrieves documents using the *rewritten* question instead of the raw one, and (4) generates the final answer grounded in that retrieved context — then appends the answer as an `AIMessage` so the checkpointer keeps the full thread.

Important components and how they work: `contextualize_question()` is the new piece — an LLM prompt whose only job is reformulating a follow-up using the prior turns, never answering it; it is a no-op on turn one (no history to resolve pronouns against). The retriever and FAISS/TF-IDF index are exactly Day 5/6's, reused unchanged. The `StateGraph`/`MessagesState`/`InMemorySaver` scaffolding is exactly Day 7's, reused unchanged. Both LLM-backed steps (contextualizing, answering) fall back to small deterministic, offline stand-ins when `GROQ_API_KEY` isn't set — the contextualization fallback uses a pronoun heuristic that prefixes the previous question — so the whole graph still runs end to end with no network call.

Why it matters: this is the wiring almost every real "chat with your documents" product uses — RAG and conversation memory are each simple on their own, but a RAG chain with no memory can't handle follow-ups, and a memory-only chat graph (Day 7) never grounds its answers in retrieved facts. Rewriting the query *before* retrieval, rather than only prompting the final answer with history, is what keeps retrieval itself on-topic once a conversation drifts into pronouns.

The practical implementation, `day8_conversational_rag.py`, runs a two-turn demo on one `thread_id` (a question about the FAISS retriever, then the vague follow-up "What about that?") and an exercise that retrieves with the raw follow-up and the rewritten standalone question side by side, reporting whether their top retrieved documents matched or differed. No new dependencies or environment variables were needed — this reuses `langgraph` (Day 7), `langchain-community` + `faiss-cpu` + `scikit-learn` (Day 5), and `langchain-groq` (Day 6), all already in `requirements.txt`.

## 20-11-2026
**Day 9: Tool Calling — binding Python functions to an LLM**

Days 6/8 grounded answers in static, pre-indexed documents via retrieval. Today's topic, **tool calling**, lets the LLM instead request that a specific Python function be run with specific arguments and read back its real result — the mechanism needed for exact computation or live actions, and the direct foundation of agents, coming up next on the roadmap.

Key pieces:

- **`langchain_core.tools.tool`** — a decorator turning a typed, documented Python function into a `StructuredTool`. Its type hints build the tool's argument schema, and its docstring becomes the natural-language description the model uses to decide when (and how) to call it.
- **`ChatGroq.bind_tools(tools)`** — attaches tool schemas to the chat model so a response may come back as an `AIMessage` with `tool_calls` populated (a list of `{"name", "args", "id"}` dicts) instead of plain text.
- **`ToolMessage`** — wraps a tool's real result, keyed to its originating call via `tool_call_id`, and gets appended to the message history before invoking the model again for a final, grounded answer.
- **The manual loop** — `invoke → if tool_calls: run them, append ToolMessages, invoke again → else: done` — exactly what agent frameworks like LangGraph's `create_react_agent` automate internally; writing it by hand once makes clear what such a framework is actually doing.

Why it matters: retrieval (Day 6/8) answers from what was indexed ahead of time; tool calling answers from what the model can compute or fetch right now, deciding per-question whether a tool is even needed — the foundation under every real agent.

Practical implementation: `day9_tool_calling.py` defines four tools (`add_numbers`, `get_word_length`, `safe_divide`, and `search_project_notes` — a deliberately tiny stand-in for wrapping Day 6's real retriever as a callable tool), binds them to `ChatGroq` when `GROQ_API_KEY` is set, and otherwise uses a deterministic offline keyword router so the full call → execute → feed-back loop still runs end-to-end with no network call. The demo covers a single-tool question, a question needing two tool calls in the same turn, and a project-note lookup; the exercise deliberately triggers a `ValueError` in `safe_divide` (division by zero) and confirms the loop catches it and reports the error back as a tool result instead of crashing. No new dependencies — `langchain_core.tools` ships with `langchain-core`, already in `requirements.txt`; no new environment variables either.
