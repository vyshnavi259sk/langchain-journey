"""
Day 8 -- Conversational RAG: making the Day 6 RAG chain history-aware
=======================================================================

Day 6 (day6_retrieval_rag.py) built a RAG chain that answers one question
well, but every `.invoke()` is independent -- a follow-up like "what k does
it use?" (where "it" means the FAISS index from the *previous* turn) gets
embedded and searched exactly as written, with no idea what "it" refers to.
Day 7 (day7_conversation_memory.py) separately solved multi-turn memory with
a LangGraph `StateGraph` + `InMemorySaver` checkpointer, but that graph
never retrieved anything -- it just chatted.

Today combines both: a **conversational RAG** graph where every turn can
(1) rewrite a context-dependent follow-up into a standalone question using
the prior conversation, (2) retrieve with that rewritten question instead
of the raw one, and (3) answer grounded in what was retrieved -- all while
the checkpointer keeps the full history, exactly like Day 7.

This is the "history-aware retriever" pattern from LangChain's own
conversational RAG tutorial, rebuilt on LangGraph (the current recommended
persistence layer, per Day 7) instead of the deprecated
`RunnableWithMessageHistory`.

Pipeline, as one graph node per turn:

    state.messages (full history so far, from the checkpointer)
        -> contextualize_question(): LLM rewrites the latest question into
           a standalone one, using the history. No rewrite needed on turn
           one (no history yet).
        -> retriever.invoke(standalone_question) -> list[Document]
        -> answer_prompt (system: "answer ONLY from this context") + llm
        -> StrOutputParser() -> AIMessage appended to state.messages

Key pieces introduced:

- **History-aware query rewriting** -- a *second*, small LLM call whose
  only job is turning a vague follow-up ("what about that?") plus the prior
  turn into a standalone question such as "What does this project's FAISS
  retriever use to build its vectors? What about that?". This is the piece
  that makes retrieval itself aware of the conversation, not just the final
  answer. Without it, step 2 above would search FAISS for the literal,
  pronoun-only follow-up, which can share little or no vocabulary with the
  TF-IDF-indexed documents -- the exercise at the bottom runs retrieval both
  ways on the same vague follow-up and reports, honestly, whether the raw
  and rewritten queries landed on the same top document or not (with only
  four toy documents in the index, a short vague query can occasionally
  still land correctly by luck; the rewritten query does not depend on
  that luck).
- **Reusing Day 6's retriever + Day 7's graph/checkpointer** -- nothing new
  is introduced at the component level; the point of today is *wiring*,
  showing how RAG and memory, covered separately, combine into the pattern
  almost every real "chat with your documents" product actually uses.

Like Day 6 and Day 7, this still runs fully offline when `GROQ_API_KEY`
is not set: the retrieval half (TF-IDF + FAISS) needs no network call at
all, and both LLM steps (contextualizing the question, generating the
final answer) fall back to small deterministic stand-ins so the whole
graph is still demonstrably end-to-end runnable with no key and no cost.

No new dependencies -- this reuses `langgraph` (Day 7), `langchain-community`
+ `faiss-cpu` + `scikit-learn` (Day 5), and `langchain-groq` (Day 6/8-1),
all already in requirements.txt. No new environment variables either.
"""

import os
import sys

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, MessagesState, StateGraph
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'langgraph'. Install with: pip install langgraph"
    ) from exc

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# 1. Day 5's offline embeddings backend, repeated here (not imported) so
#    this file reads and runs standalone, same convention as day6/day7.
# ---------------------------------------------------------------------------
class TfidfEmbeddings(Embeddings):
    """Minimal, local, offline Embeddings backend -- no API key, no network."""

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
            raise RuntimeError("TfidfEmbeddings must be fitted before embedding a query.")
        vector = self._vectorizer.transform([text])
        return vector.toarray().astype(float)[0].tolist()


# ---------------------------------------------------------------------------
# 2. A small knowledge base, deliberately containing details a follow-up
#    like "what k does IT use?" can only resolve correctly if the retrieval
#    query is first rewritten to mention "FAISS retriever" explicitly.
# ---------------------------------------------------------------------------
def build_knowledge_base() -> list[Document]:
    raw_notes = [
        (
            "This project's FAISS retriever (built in day5_embeddings_vectorstore.py "
            "and reused in day6_retrieval_rag.py) uses k=2 nearest neighbors by "
            "default when wrapped with .as_retriever(search_kwargs={'k': 2}).",
            "project-config",
        ),
        (
            "The FAISS index is built from TF-IDF vectors produced by scikit-learn's "
            "TfidfVectorizer, not a neural embedding model, so it needs no API key "
            "and no network call to build or query.",
            "project-config",
        ),
        (
            "Day 7 of this project (day7_conversation_memory.py) replaced the "
            "deprecated RunnableWithMessageHistory with a LangGraph StateGraph "
            "plus an InMemorySaver checkpointer keyed by thread_id.",
            "roadmap-notes",
        ),
        (
            "Day 8 of this project (day8_conversational_rag.py) adds a history-aware "
            "question-rewriting step before retrieval, so follow-up questions that "
            "use pronouns like 'it' or 'that' can still be retrieved correctly.",
            "roadmap-notes",
        ),
    ]
    return [Document(page_content=text, metadata={"source": source}) for text, source in raw_notes]


def build_retriever(k: int = 2):
    documents = build_knowledge_base()
    embeddings = TfidfEmbeddings()
    embeddings.fit([doc.page_content for doc in documents])
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"- ({doc.metadata['source']}) {doc.page_content}" for doc in docs)


# ---------------------------------------------------------------------------
# 3. The LLM-backed steps, each with a free/offline fallback so the whole
#    graph runs with or without GROQ_API_KEY -- same convention as Day 7.
# ---------------------------------------------------------------------------
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, rewrite it as a standalone question that can "
    "be understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed, and otherwise return it unchanged. Return ONLY "
    "the question, with no preamble."
)

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below. If the context doesn't "
    "contain the answer, say you don't know rather than guessing.\n\nContext:\n{context}"
)


def _get_llm():
    """Lazily builds a ChatGroq instance, or returns None if no key is set.
    Isolated here so both LLM-backed steps below share one failure path."""
    if not GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
    except Exception as exc:  # bad key, network error, package missing, etc.
        print(f"(Could not construct ChatGroq: {exc!r} -- falling back to offline stand-ins.)")
        return None


def contextualize_question(history: list[BaseMessage], latest_question: str) -> str:
    """Rewrite `latest_question` into a standalone question, using `history`.
    Returns the question unchanged when there is no prior history (turn one
    has nothing to resolve pronouns against)."""
    if not history:
        return latest_question

    llm = _get_llm()
    if llm is not None:
        try:
            messages = [
                SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
                *history,
                HumanMessage(content=latest_question),
            ]
            result = llm.invoke(messages)
            rewritten = (result.content or "").strip()
            if rewritten:
                return rewritten
        except Exception as exc:  # network/auth failure -- fall through to heuristic
            print(f"(Live contextualization call failed: {exc!r} -- using offline heuristic.)")

    # Offline / no-API-key fallback: a crude but free heuristic. If the
    # latest question leans on a pronoun and there IS prior history, prefix
    # it with the previous human turn so the retrieval query at least shares
    # vocabulary with the earlier question instead of standing completely
    # alone. This is not a substitute for a real LLM rewrite -- it exists so
    # the graph is still fully runnable, end to end, with no network call.
    pronouns = ("it", "that", "this", "those", "these", "they")
    needs_context = any(word.strip(",.?!").lower() in pronouns for word in latest_question.split())
    if needs_context:
        previous_human = next(
            (m.content for m in reversed(history) if isinstance(m, HumanMessage)),
            "",
        )
        if previous_human:
            return f"{previous_human} {latest_question}"
    return latest_question


def generate_answer(question: str, context: str) -> str:
    """Generate the final grounded answer, or an offline stand-in."""
    llm = _get_llm()
    if llm is not None:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", ANSWER_SYSTEM_PROMPT), ("user", "{question}")]
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        except Exception as exc:  # network/auth failure -- fall through to offline view
            print(f"(Live answer-generation call failed: {exc!r} -- falling back to offline view.)")

    return (
        "[offline stand-in -- set GROQ_API_KEY for a real grounded answer]\n"
        f"  Standalone question used for retrieval: {question!r}\n"
        f"  Retrieved context:\n{context}"
    )


# ---------------------------------------------------------------------------
# 4. The LangGraph node and app: one node per turn, same shape as Day 7,
#    but now it retrieves before it answers.
# ---------------------------------------------------------------------------
def make_conversational_rag_node(retriever):
    """Returns a graph node closed over a specific retriever, so the demo
    and the exercise below can each build the graph against a different k."""

    def node(state: MessagesState) -> dict:
        history = state["messages"][:-1]  # everything before this turn's question
        latest_question = state["messages"][-1].content

        standalone_question = contextualize_question(history, latest_question)
        docs = retriever.invoke(standalone_question)
        context = format_docs(docs)
        answer = generate_answer(standalone_question, context)

        return {"messages": [AIMessage(content=answer)]}

    return node


def build_app(retriever):
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("conversational_rag", make_conversational_rag_node(retriever))
    workflow.add_edge(START, "conversational_rag")
    return workflow.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 5. Demo: a follow-up question that only makes sense once "it" is resolved
#    against the previous turn.
# ---------------------------------------------------------------------------
def run_demo() -> None:
    retriever = build_retriever(k=2)
    app = build_app(retriever)
    thread = {"configurable": {"thread_id": "conversational-rag-demo"}}

    print("--- Turn 1 ---")
    q1 = "What does this project's FAISS retriever use to build its vectors?"
    out1 = app.invoke({"messages": [HumanMessage(content=q1)]}, config=thread)
    print(f"Q: {q1}\nA: {out1['messages'][-1].content}")

    print("\n--- Turn 2 (follow-up: 'it' refers to the retriever from turn 1) ---")
    q2 = "What about that?"
    out2 = app.invoke({"messages": [HumanMessage(content=q2)]}, config=thread)
    print(f"Q: {q2}\nA: {out2['messages'][-1].content}")
    print(f"(thread now holds {len(out2['messages'])} total messages)")

    snapshot = app.get_state(thread)
    print(f"\nCheckpointer confirms the persisted thread has {len(snapshot.values['messages'])} messages.")


# ---------------------------------------------------------------------------
# 6. Exercise: show WHY the rewrite step matters, by retrieving with the
#    raw follow-up question instead of the standalone, rewritten one.
# ---------------------------------------------------------------------------
def exercise_contextualization_matters() -> None:
    print("\n=== Exercise: retrieval with vs. without query rewriting ===")
    retriever = build_retriever(k=1)

    history = [
        HumanMessage(content="What does this project's FAISS retriever use to build its vectors?"),
        AIMessage(content="TF-IDF vectors from scikit-learn's TfidfVectorizer."),
    ]
    follow_up = "What about that?"

    print(f"\nRaw follow-up question used directly as the retrieval query: {follow_up!r}")
    raw_docs = retriever.invoke(follow_up)
    for doc in raw_docs:
        print(f"  - ({doc.metadata['source']}) {doc.page_content[:90]}...")

    standalone = contextualize_question(history, follow_up)
    print(f"\nRewritten, standalone question used instead: {standalone!r}")
    rewritten_docs = retriever.invoke(standalone)
    for doc in rewritten_docs:
        print(f"  - ({doc.metadata['source']}) {doc.page_content[:90]}...")

    same_top_doc = raw_docs and rewritten_docs and raw_docs[0].metadata == rewritten_docs[0].metadata
    print(
        "\n(Top retrieved document "
        + ("matched" if same_top_doc else "DIFFERED")
        + " between the raw and rewritten query -- "
        "rewriting the follow-up to be standalone is what keeps retrieval on-topic "
        "once pronouns start referring back to earlier turns.)"
    )


if __name__ == "__main__":
    try:
        run_demo()
        exercise_contextualization_matters()
    except Exception as exc:  # top-level guard so the script fails loudly, not silently
        print(f"Unexpected error while running the conversational RAG demo: {exc}")
        sys.exit(1)
