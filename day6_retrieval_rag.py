"""
Day 6: Retrieval-Augmented Generation (RAG) -- wiring a retriever into a chain
================================================================================

Day 5 (day5_embeddings_vectorstore.py) built a FAISS vector store and showed
that `.as_retriever()` returns a standard LangChain `Runnable`. Day 2
(day2_lcel_runnables.py) showed `RunnableParallel` fanning one input into
several branches and merging the results into a dict. Today's topic is what
those two facts are *for*: putting them together into a real **RAG chain**.

The RAG pattern, as a single LCEL expression:

    rag_chain = (
        RunnableParallel(context=retriever, question=RunnablePassthrough())
        | prompt
        | llm
        | StrOutputParser()
    )

Reading it left to right:

1. `RunnableParallel(context=retriever, question=RunnablePassthrough())`
   takes the raw input string (the user's question) and runs two branches
   on it at once: `retriever` turns the question into a list of relevant
   `Document`s (the "context"), while `RunnablePassthrough()` just forwards
   the original question unchanged. The two results are merged into
   `{"context": [...], "question": "..."}`.
2. A small `RunnableLambda` (`format_docs`) joins the retrieved Documents
   into one string before they reach the prompt -- a prompt template wants
   text, not a list of Document objects.
3. `prompt` is a `ChatPromptTemplate` with a system instruction telling the
   model to answer *only* from the provided context, plus the merged
   `context`/`question` dict filling in the human message.
4. `llm` (ChatGroq, same model used in day6.py/day7.py/day8-*.py) generates
   a response grounded in that retrieved context instead of relying purely
   on what it memorized during training -- this is the "G" in RAG.
5. `StrOutputParser()` unwraps the `AIMessage` into a plain string, exactly
   like day8-1.py.

Why this matters: an LLM's training data is frozen and it has no idea about
documents that live only in *your* vector store. A RAG chain fixes that at
query time -- retrieve the most relevant chunks, stuff them into the prompt,
then ask the model to answer *using* them. This is the same general shape
used by every production "chat with your docs" system, just with a FAISS
index this small as the toy corpus.

This file reuses Day 5's `TfidfEmbeddings` + FAISS index so the retrieval
half runs fully offline. If `GROQ_API_KEY` is not set, the generation half
is skipped and the script prints the retrieved context plus the exact
prompt that *would* have been sent, so the LCEL wiring is still fully
demonstrated without a network call.

Next topic in the roadmap: conversation memory / chat history, so a RAG
chain can take follow-up questions ("what about the second one?") instead
of treating every query in isolation.
"""

import os
import sys

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'scikit-learn'. Install with: pip install scikit-learn"
    ) from exc

try:
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'faiss-cpu' and 'langchain-community'. "
        "Install with: pip install faiss-cpu langchain-community"
    ) from exc

load_dotenv()


class TfidfEmbeddings(Embeddings):
    """Same minimal, local, offline Embeddings backend as Day 5 -- repeated
    here (rather than imported from day5_embeddings_vectorstore.py) so this
    file can be read and run on its own, the same way day6.py/day7.py stand
    alone rather than importing from each other."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Cannot fit TfidfEmbeddings on an empty corpus.")
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError(
                "TfidfEmbeddings must be fitted before embedding a query."
            )
        vector = self._vectorizer.transform([text])
        return vector.toarray().astype(float)[0].tolist()


def build_knowledge_base() -> list[Document]:
    """A tiny corpus of LangChain fact-sheets, deliberately covering things
    a general-purpose LLM would NOT reliably know -- the fictional/explicit
    details below -- so a correct RAG answer can only come from retrieval,
    not from the model's own training data."""
    raw_notes = [
        (
            "The LangChain Journey project's FAISS index is configured with "
            "k=3 nearest neighbors by default and uses TF-IDF vectors, not "
            "neural embeddings, because no paid embeddings API is required "
            "for this learning journey.",
            "project-config",
        ),
        (
            "In this project's roadmap, Day 6 introduces Retrieval-Augmented "
            "Generation by combining a retriever with RunnableParallel and "
            "RunnablePassthrough from day2_lcel_runnables.py.",
            "roadmap-notes",
        ),
        (
            "Document loaders (day3_document_loaders.py) and text splitting "
            "(day4_text_splitting.py) both run upstream of embeddings -- "
            "raw files become Documents, Documents become chunks, chunks "
            "become vectors.",
            "roadmap-notes",
        ),
        (
            "This project's chat model is ChatGroq running "
            "llama-3.1-8b-instant, loaded with an API key read from the "
            "GROQ_API_KEY environment variable via python-dotenv.",
            "project-config",
        ),
    ]
    return [
        Document(page_content=text, metadata={"source": source})
        for text, source in raw_notes
    ]


def build_retriever(k: int = 2):
    """Builds the Day 5-style FAISS index and returns it as a Runnable
    retriever -- the exact object `RunnableParallel` below treats as just
    another branch."""
    documents = build_knowledge_base()
    embeddings = TfidfEmbeddings()
    embeddings.fit([doc.page_content for doc in documents])
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """RunnableLambda target: turns a list of retrieved Documents into one
    string a prompt template can slot in as {context}."""
    return "\n\n".join(f"- ({doc.metadata['source']}) {doc.page_content}" for doc in docs)


def build_rag_chain(retriever):
    """Assembles the full LCEL RAG pipeline described in the module
    docstring and returns it, unexecuted, so callers can `.invoke()` it."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using ONLY the context below. "
                "If the context doesn't contain the answer, say you don't know "
                "rather than guessing.\n\nContext:\n{context}",
            ),
            ("user", "{question}"),
        ]
    )

    retrieval_stage = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough(),
    )

    # Importing ChatGroq lazily so the retrieval half of this file can still
    # be exercised even in environments where langchain-groq isn't installed.
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key available -- the retrieval + prompt-assembly half of the
        # chain is still fully usable without one; the LLM stage just isn't
        # constructed. run_rag_demo() falls back to inspecting those two
        # stages directly in this case.
        return None, retrieval_stage, prompt

    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")

    return retrieval_stage | prompt | llm | StrOutputParser(), retrieval_stage, prompt


def run_rag_demo() -> None:
    retriever = build_retriever(k=2)

    questions = [
        "How many nearest neighbors does this project's FAISS retriever use by default?",
        "What two Runnables from day2 does Day 6 combine to build the RAG chain?",
    ]

    rag_chain, retrieval_stage, prompt = build_rag_chain(retriever)

    for question in questions:
        print(f"\nQuestion: {question!r}")

        if rag_chain is not None:
            try:
                answer = rag_chain.invoke(question)
                print(f"Answer: {answer}")
                continue
            except Exception as exc:  # network/auth failures, surfaced not swallowed
                print(f"  (LLM call failed, falling back to retrieval-only view: {exc})")

        # No API key (or the call failed) -- still demonstrate the retrieval
        # + prompt-assembly half of the chain, which needs no network call.
        merged = retrieval_stage.invoke(question)
        rendered_prompt = prompt.invoke(merged)
        print("  [No working GROQ_API_KEY/OPENAI_API_KEY -- showing retrieval + "
              "prompt assembly only]")
        print(f"  Retrieved context:\n{merged['context']}")
        print(f"  Rendered prompt messages: {rendered_prompt.to_messages()}")

    exercise_compare_k(retriever_builder=build_retriever)


def exercise_compare_k(retriever_builder) -> None:
    """Exercise: show how changing k (the number of retrieved chunks) can
    change what context the chain has to work with -- the classic
    precision/recall tradeoff in retrieval."""
    print("\n=== Exercise: effect of k on retrieved context ===")
    query = "What model and provider does this project's chat LLM use?"
    for k in (1, 3):
        retriever = retriever_builder(k=k)
        docs = retriever.invoke(query)
        print(f"\nk={k} -> {len(docs)} document(s) retrieved:")
        for doc in docs:
            print(f"  - ({doc.metadata['source']}) {doc.page_content[:80]}...")


if __name__ == "__main__":
    try:
        run_rag_demo()
    except Exception as exc:  # top-level guard so the script fails loudly, not silently
        print(f"Unexpected error while running the RAG demo: {exc}")
        sys.exit(1)
