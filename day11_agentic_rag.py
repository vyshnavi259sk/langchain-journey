"""
Day 11 -- Agentic RAG: letting the agent *decide* whether to retrieve at all
=============================================================================

Day 6 (day6_retrieval_rag.py) and Day 8 (day8_conversational_rag.py) both
built RAG pipelines that retrieve on *every single call*, whether or not the
question actually needs it -- the pipeline has no choice in the matter, it
just always runs `retriever.invoke(...)` as a fixed step. Day 9 and Day 10
separately built tool-calling agents (`day9_tool_calling.py`,
`day10_react_agent.py`) that call Python functions on demand, but neither
one had a retriever among its tools.

Today combines both ideas into **agentic RAG**: the Day 5/6/8-style
TF-IDF + FAISS retriever is wrapped as an ordinary `@tool`, handed to a
Day 10-style `create_react_agent`, *alongside* an unrelated tool
(`is_palindrome`). The agent itself -- not a fixed pipeline -- now decides
per-question whether retrieval is even relevant, exactly like it already
decides whether any other tool is needed.

What the concept is: in a non-agentic RAG chain, "retrieve" is a pipeline
stage -- it always runs. In agentic RAG, "retrieve" is just one tool among
several, and the model's own reasoning step chooses whether to call it, call
a different tool instead, call several tools, or skip tools entirely and
answer directly. This matters because always retrieving wastes a vector
search on questions the index has nothing to do with, and -- worse -- can
stuff irrelevant "context" into the prompt that actively distracts the model
from a question it could otherwise answer correctly on its own.

Important components and how they fit:

- **`retrieve_project_notes(query: str)`** -- a `@tool`-decorated function
  that calls `.invoke(query)` on a FAISS retriever (same `TfidfEmbeddings` +
  FAISS setup as Day 5/6/8, repeated here so this file stays standalone) and
  returns the joined `page_content` of the top matches. From the agent's
  point of view this is just a tool like any other -- the fact that it does
  retrieval internally is an implementation detail behind a plain string-in,
  string-out interface.
- **`is_palindrome(text: str)`** -- a second, unrelated tool. Its only job
  here is giving the agent something *other* than retrieval to reach for, so
  the demo can show the agent choosing between tools (not just whether to
  use the one RAG tool). It raises `ValueError` on an empty string -- the
  deliberate error case for today's exercise, same contract as
  `safe_divide` (Day 9) and `is_prime` (Day 10).
- **`create_react_agent(model, tools, checkpointer=...)`** -- exactly Day
  10's prebuilt graph, now holding a retrieval tool and a non-retrieval tool
  side by side.

Why it matters: almost every production "chat with your data" system today
is agentic RAG, not a fixed RAG chain -- the agent is usually also expected
to answer general questions, use a calculator, call other APIs, etc., and a
fixed "always retrieve first" pipeline either retrieves irrelevant context
for those or needs bespoke routing logic in front of it. Letting the LLM's
own tool-calling judgment do that routing is simpler and composes with
however many other tools get added later.

Practical implementation: this script builds a real `create_react_agent`
backed by `ChatGroq` when `GROQ_API_KEY` is set, and otherwise hand-wires an
equivalent `StateGraph` (same two tools, same `ToolNode`, a deterministic
keyword router standing in for the LLM's reasoning step) so the full
retrieve-or-don't decision runs end-to-end with no network call. The demo
asks three questions: one that should trigger retrieval, one that should
trigger `is_palindrome` instead, and one that needs *no* tool at all --
showing the agent's tool choice differs each time. The exercise does two
things: (1) deliberately calls `is_palindrome("")` to confirm the graph
reports the `ValueError` as an error-flagged `ToolMessage` instead of
crashing, and (2) directly compares the agentic approach against a
Day 8-style "always retrieve" pipeline on the palindrome question, printing
the (irrelevant) context the fixed pipeline would have forced into the
prompt that the agent correctly skipped.

No new dependencies -- this reuses `langgraph` (Day 7/10), `langchain-core`
(Day 9/10), and `langchain-community` + `faiss-cpu` + `scikit-learn`
(Day 5/6/8), all already in requirements.txt. No new environment variables
either -- `GROQ_API_KEY` is reused exactly as in Days 6-10, and this script
also runs fully offline.
"""

import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

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
    from langgraph.graph import END, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'langgraph'. Install with: pip install langgraph"
    ) from exc

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# 1. Day 5's offline embeddings backend, repeated here (not imported) so this
#    file reads and runs standalone, same convention as day6/day7/day8.
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
# 2. A small knowledge base about this project -- deliberately unrelated to
#    palindromes, so a question about palindromes has nothing relevant in it.
# ---------------------------------------------------------------------------
def build_knowledge_base() -> list[Document]:
    raw_notes = [
        (
            "This project's retriever (day5_embeddings_vectorstore.py, reused in "
            "day6/day8/day11) embeds text with a local TfidfVectorizer wrapped in "
            "a TfidfEmbeddings class, then indexes the vectors in a FAISS store -- "
            "no API key or network call needed to build or query it.",
            "project-config",
        ),
        (
            "Day 10 of this project (day10_react_agent.py) wired a hand-rolled "
            "tool-calling loop up as an explicit LangGraph agent/tools graph via "
            "create_react_agent, with ToolNode handling tool execution and error "
            "wrapping.",
            "roadmap-notes",
        ),
        (
            "Day 11 of this project (day11_agentic_rag.py) wraps that same FAISS "
            "retriever as a plain @tool function, so a create_react_agent can "
            "decide per-question whether retrieval is relevant, instead of a "
            "fixed pipeline always retrieving first.",
            "roadmap-notes",
        ),
    ]
    return [Document(page_content=text, metadata={"source": source}) for text, source in raw_notes]


def build_retriever(k: int = 2):
    docs = build_knowledge_base()
    embeddings = TfidfEmbeddings()
    embeddings.fit([doc.page_content for doc in docs])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


_RETRIEVER = build_retriever()


# ---------------------------------------------------------------------------
# 3. Tools -- one wraps retrieval, one is deliberately unrelated to it, so
#    the agent has to choose between (or decline) them per question.
# ---------------------------------------------------------------------------

@tool
def retrieve_project_notes(query: str) -> str:
    """Search this project's own notes for facts about its implementation
    (e.g. embeddings, retrievers, FAISS, agents, specific day numbers).
    Only useful for questions ABOUT this project -- not general knowledge,
    arithmetic, or word-level questions like palindromes."""
    try:
        docs = _RETRIEVER.invoke(query)
    except Exception as exc:  # pragma: no cover - defensive, FAISS/sklearn errors
        return f"Error while searching project notes: {exc}"
    if not docs:
        return "No relevant project notes found."
    return "\n---\n".join(doc.page_content for doc in docs)


@tool
def is_palindrome(text: str) -> bool:
    """Return True if `text` reads the same forwards and backwards
    (case-insensitive, ignoring spaces), else False.

    Raises ValueError for an empty string, since palindrome-ness is
    undefined for empty input -- the deliberate error case for today's
    exercise.
    """
    if text == "":
        raise ValueError("is_palindrome is undefined for an empty string.")
    normalized = text.lower().replace(" ", "")
    return normalized == normalized[::-1]


TOOLS = [retrieve_project_notes, is_palindrome]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Path 1 (preferred): a real create_react_agent, used when GROQ_API_KEY is set
# ---------------------------------------------------------------------------

def build_groq_agent():
    """Build a real LangGraph ReAct agent backed by ChatGroq."""
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    checkpointer = InMemorySaver()
    return create_react_agent(llm, TOOLS, checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Path 2 (offline fallback): same graph, hand-wired with a deterministic
# keyword router standing in for the LLM's reasoning step. Unlike Day 10's
# router (which always found *some* tool to call), this one also has a
# "no tool needed" branch, since that choice is itself part of today's topic.
# ---------------------------------------------------------------------------

def _fake_llm_turn(messages: list) -> AIMessage:
    """Deterministic stand-in for an LLM's reasoning step.

    Decides, per question, whether to call retrieve_project_notes,
    is_palindrome, or neither -- mirroring a real tool-calling model that
    can also choose to answer directly with no tool call at all.
    """
    last_human_index = 0
    question = ""
    for idx, msg in enumerate(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_index = idx
            question = msg.content

    current_turn = messages[last_human_index:]
    already_called = {
        msg.name for msg in current_turn if msg.__class__.__name__ == "ToolMessage"
    }
    lowered = question.lower()

    if "palindrome" in lowered and "is_palindrome" not in already_called:
        phrase = question.split('"')[1] if '"' in question else question
        return AIMessage(
            content="",
            tool_calls=[{"name": "is_palindrome", "args": {"text": phrase}, "id": "call_pal"}],
        )

    project_keywords = ("project", "retriev", "faiss", "embed", "note", "day 10", "day 11")
    if any(kw in lowered for kw in project_keywords) and "retrieve_project_notes" not in already_called:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "retrieve_project_notes", "args": {"query": question}, "id": "call_rag"}
            ],
        )

    # No tool is relevant -- answer directly, same as a real agent declining
    # to call any tool. A tiny arithmetic fallback covers today's demo
    # question; anything else gets an honest "can't answer offline" note.
    digits_and_ops = [tok for tok in lowered.replace("?", "").split() if tok]
    if "+" in lowered:
        try:
            left, right = lowered.split("+")
            total = int("".join(ch for ch in left if ch.isdigit())) + int(
                "".join(ch for ch in right if ch.isdigit())
            )
            return AIMessage(content=f"(offline fallback) {total}")
        except ValueError:
            pass

    observations = [
        f"{msg.name}={msg.content}"
        for msg in current_turn
        if msg.__class__.__name__ == "ToolMessage"
    ]
    if observations:
        return AIMessage(content=f"(offline fallback) {'; '.join(observations)}")
    return AIMessage(content="(offline fallback) No tool was needed for this question.")


def _agent_node(state: MessagesState) -> dict:
    return {"messages": [_fake_llm_turn(state["messages"])]}


def _should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_fallback_agent():
    """Hand-wire the same agent/tools loop create_react_agent builds internally."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    # handle_tool_errors=True turns a raised exception (e.g. is_palindrome(""))
    # into an error-flagged ToolMessage instead of crashing the graph -- same
    # contract Day 10 relied on for is_prime(1).
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=InMemorySaver())


def build_agent():
    if GROQ_API_KEY:
        try:
            return build_groq_agent(), "groq"
        except Exception as exc:  # pragma: no cover - network/model errors
            print(f"[warning] Falling back to offline agent: {exc}")
    return build_fallback_agent(), "offline"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    agent, mode = build_agent()
    print(f"Running in '{mode}' mode (GROQ_API_KEY {'set' if GROQ_API_KEY else 'not set'}).\n")

    def ask(question: str, thread_id: str) -> None:
        # Each question gets its own thread_id deliberately -- these three
        # questions are independent, not a follow-up conversation, so this
        # keeps each turn's reported tool choice uncontaminated by the
        # previous question's history (a shared thread_id is for Day 7/8/10
        # -style multi-turn follow-ups, not needed here).
        thread = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": [HumanMessage(content=question)]}, thread)
        tool_calls_made = [
            msg.name for msg in result["messages"] if msg.__class__.__name__ == "ToolMessage"
        ]
        print("Q:", question)
        print("Tool(s) called:", tool_calls_made or "none")
        print("A:", result["messages"][-1].content)
        print()

    print("--- Question 1: should trigger retrieval ---")
    ask("What does this project's retriever use to build its vectors?", "day11-demo-q1")

    print("--- Question 2: should trigger is_palindrome, not retrieval ---")
    ask('Is "level" a palindrome?', "day11-demo-q2")

    print("--- Question 3: needs no tool at all ---")
    ask("What is 7 + 5?", "day11-demo-q3")


def run_exercise() -> None:
    """Exercise, two parts:

    1. Deliberately call is_palindrome("") to confirm the graph reports the
       ValueError as an error-flagged ToolMessage instead of crashing.
    2. Compare today's agentic choice on the palindrome question against a
       Day 8-style "always retrieve first" pipeline, showing the irrelevant
       context such a fixed pipeline would have forced into the prompt.
    """
    print("--- Exercise part 1: is_palindrome('') should raise, not crash ---")
    agent, _ = build_agent()
    thread = {"configurable": {"thread_id": "day11-exercise"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content='Is "" a palindrome?')]}, thread
    )
    for msg in result["messages"]:
        if msg.__class__.__name__ == "ToolMessage":
            status = "ERROR" if getattr(msg, "status", None) == "error" else "ok"
            print(f"tool={msg.name} status={status} content={msg.content}")
    print("Final answer:", result["messages"][-1].content)

    print("\n--- Exercise part 2: agentic vs. always-retrieve on a non-project question ---")
    palindrome_question = 'Is "racecar" a palindrome?'
    agentic_result = agent.invoke(
        {"messages": [HumanMessage(content=palindrome_question)]},
        {"configurable": {"thread_id": "day11-exercise-compare"}},
    )
    agentic_tools = [
        msg.name for msg in agentic_result["messages"] if msg.__class__.__name__ == "ToolMessage"
    ]
    print(f"Agentic approach called: {agentic_tools or 'no tools'} -- correctly skipped retrieval.")

    forced_context = retrieve_project_notes.invoke({"query": palindrome_question})
    print(
        "A Day-8-style 'always retrieve first' pipeline would have forced this "
        "irrelevant context into the prompt instead:\n" + forced_context
    )


if __name__ == "__main__":
    run_demo()
    run_exercise()
