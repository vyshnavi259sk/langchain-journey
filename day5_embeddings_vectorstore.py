"""
Day 5: Embeddings and Vector Stores
=====================================

Day 4 (day4_text_splitting.py) ended with a stack of small text chunks that
still had `metadata["source"]` attached. Those chunks are useless for
*semantic* search until each one is turned into a vector -- a fixed-length
list of floats that places similar meanings close together in space. That
conversion step is an **embedding model**, and the structure that stores the
resulting vectors (and lets you search them by similarity) is a
**vector store**.

Today covers both halves of that pipeline using LangChain's real interfaces:

- `langchain_core.embeddings.Embeddings` -- the abstract base class every
  embedding backend implements. It only needs two methods:
    * `embed_documents(texts: list[str]) -> list[list[float]]`
    * `embed_query(text: str) -> list[float]`
  OpenAIEmbeddings, HuggingFaceEmbeddings, etc. are all just implementations
  of this same interface, which is *why* a vector store built against one
  embedding backend can be swapped to another without touching the rest of
  the chain.

- `langchain_community.vectorstores.FAISS` -- a LangChain wrapper around
  Facebook AI Similarity Search (a local, in-memory vector index -- no
  server, no API key). It exposes `.from_documents()`, `.similarity_search()`,
  `.similarity_search_with_score()`, and `.as_retriever()`.

Why a *custom* Embeddings class today instead of an API-based one: Groq (used
in day6.py/day7.py/day8-*.py for chat) does not serve an embeddings endpoint,
and a real sentence-transformer model would pull in a multi-hundred-MB torch
download. So this file implements `TfidfEmbeddings`, a small local embedding
backend built on scikit-learn's `TfidfVectorizer`. It is a legitimate,
free/local embedding method (classic information retrieval, no neural net,
no API key, no network call) -- good enough to demonstrate real nearest-
neighbor retrieval behavior, while keeping the exercise fully offline and
fast. The same `TfidfEmbeddings` class is a drop-in `Embeddings` subclass:
everything below it (FAISS, retriever, similarity scores) works completely
unchanged with `OpenAIEmbeddings` or `HuggingFaceEmbeddings` if you swap the
import and provide an API key via `os.getenv(...)` -- see the commented
example near the bottom.

Next topic in the roadmap: retrieval chains / RAG, which plugs this
vector store's `.as_retriever()` straight into the `RunnableParallel`
pattern from day2_lcel_runnables.py (`{"context": retriever, "question":
RunnablePassthrough()}`).
"""

import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'faiss-cpu' and 'langchain-community'. "
        "Install with: pip install faiss-cpu langchain-community"
    ) from exc


class TfidfEmbeddings(Embeddings):
    """A minimal, local, offline Embeddings backend.

    Implements the same two methods every LangChain embedding class
    implements. Internally it fits a scikit-learn TF-IDF vectorizer on the
    corpus it first sees, then reuses that fitted vocabulary for every later
    call -- mirroring how a *real* embedding model has a fixed vector space
    once trained.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the TF-IDF vocabulary on a corpus. Must run before embedding."""
        if not corpus:
            raise ValueError("Cannot fit TfidfEmbeddings on an empty corpus.")
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            # Fit lazily on first batch if the caller skipped an explicit fit.
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbeddings must be fitted (via embed_documents or "
                ".fit()) before embedding a query."
            )
        vector = self._vectorizer.transform([text])
        return vector.toarray().astype(float)[0].tolist()


def build_sample_knowledge_base() -> list[Document]:
    """A tiny corpus about this same learning journey, so similarity search
    results are easy to sanity-check by eye."""
    raw_notes = [
        (
            "Document loaders in LangChain read raw sources -- text files, "
            "PDFs, web pages -- into a list of Document objects with "
            "page_content and metadata.",
            "day3_document_loaders.py",
        ),
        (
            "RecursiveCharacterTextSplitter breaks long documents into "
            "overlapping chunks using chunk_size and chunk_overlap, trying "
            "paragraph, line, and word boundaries before a hard cut.",
            "day4_text_splitting.py",
        ),
        (
            "RunnableParallel runs several branches on the same input and "
            "merges the results into a dict, which is exactly the shape a "
            "retrieval chain needs: context from a retriever plus the "
            "original question.",
            "day2_lcel_runnables.py",
        ),
        (
            "A vector store indexes embeddings so that semantically similar "
            "text can be found with a similarity search instead of exact "
            "keyword matching.",
            "day5_embeddings_vectorstore.py",
        ),
        (
            "ChatPromptTemplate combines a system instruction with "
            "user-supplied variables, and piping it into a chat model with "
            "the | operator builds an LCEL chain.",
            "day7.py",
        ),
        (
            "StrOutputParser converts a chat model's AIMessage response "
            "into a plain string, which is the typical last step of an "
            "LCEL chain.",
            "day8-1.py",
        ),
    ]
    return [
        Document(page_content=text, metadata={"source": source})
        for text, source in raw_notes
    ]


def run_similarity_search_demo() -> None:
    documents = build_sample_knowledge_base()

    embeddings = TfidfEmbeddings()
    # Fit the vocabulary on the full corpus up front, then build the index.
    embeddings.fit([doc.page_content for doc in documents])

    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
    except Exception as exc:  # defensive: surfaces faiss/env issues clearly
        print(f"Failed to build the FAISS vector store: {exc}")
        sys.exit(1)

    queries = [
        "How do I split a long document into smaller pieces?",
        "How do I turn a model's reply into a plain string?",
    ]

    print("=== similarity_search ===")
    for query in queries:
        print(f"\nQuery: {query!r}")
        results = vectorstore.similarity_search(query, k=2)
        for rank, doc in enumerate(results, start=1):
            print(f"  {rank}. [{doc.metadata['source']}] {doc.page_content}")

    print("\n=== similarity_search_with_score (lower = closer, FAISS L2) ===")
    query = "What merges several parallel branches into one dict?"
    print(f"Query: {query!r}")
    for doc, score in vectorstore.similarity_search_with_score(query, k=3):
        print(f"  score={score:.4f} [{doc.metadata['source']}] {doc.page_content[:70]}...")

    # as_retriever() wraps the vector store in LangChain's standard Runnable
    # interface, so it can sit inside an LCEL chain exactly like the
    # RunnableLambda/RunnableParallel branches from day2_lcel_runnables.py.
    print("\n=== as_retriever() (Runnable interface) ===")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    retrieved = retriever.invoke("What does a document loader return?")
    for doc in retrieved:
        print(f"  [{doc.metadata['source']}] {doc.page_content}")

    exercise_add_and_requery(vectorstore)


def exercise_add_and_requery(vectorstore: "FAISS") -> None:
    """Exercise: add a brand-new document to an already-built vector store
    and confirm it is retrievable without rebuilding the whole index."""
    print("\n=== Exercise: add_documents() then re-query ===")
    new_doc = Document(
        page_content=(
            "FAISS.add_documents() lets you grow an existing vector store "
            "incrementally, embedding and indexing only the new documents "
            "instead of rebuilding from scratch."
        ),
        metadata={"source": "day5_embeddings_vectorstore.py (exercise)"},
    )
    try:
        vectorstore.add_documents([new_doc])
    except Exception as exc:
        print(f"  add_documents failed: {exc}")
        return

    query = "Can I add new documents to a vector store after it's built?"
    results = vectorstore.similarity_search(query, k=1)
    print(f"Query: {query!r}")
    for doc in results:
        print(f"  -> [{doc.metadata['source']}] {doc.page_content}")


# ---------------------------------------------------------------------------
# Swapping in a real API-based embedding model (not run here -- for reference):
#
#   import os
#   from langchain_openai import OpenAIEmbeddings
#   api_key = os.getenv("OPENAI_API_KEY")
#   if not api_key:
#       raise RuntimeError("Set OPENAI_API_KEY in your .env to use OpenAIEmbeddings.")
#   embeddings = OpenAIEmbeddings(api_key=api_key)
#
# Everything from `FAISS.from_documents(documents, embeddings)` onward is
# identical -- that's the whole point of the shared Embeddings interface.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        run_similarity_search_demo()
    except Exception as exc:  # top-level guard so the script fails loudly, not silently
        print(f"Unexpected error while running the embeddings/vector store demo: {exc}")
        sys.exit(1)
