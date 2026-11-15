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
