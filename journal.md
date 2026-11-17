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
