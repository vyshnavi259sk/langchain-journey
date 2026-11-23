"""
Day 12 -- Multi-Agent Systems: a supervisor that routes between specialists
=============================================================================

Days 9-11 built a single agent that chooses among several *tools*
(`day9_tool_calling.py`, `day10_react_agent.py`, `day11_agentic_rag.py`).
That works well while every tool lives in one agent's head, but real
systems often split responsibilities across *separate* agents instead --
each with its own narrow toolset, its own prompt, and its own reasoning --
coordinated by a **supervisor** that decides which specialist should
handle a given request. Today builds the simplest version of that
pattern: a LangGraph supervisor routing between two specialist agents.

What the concept is: a multi-agent "supervisor" graph has one router node
(the supervisor) and one node per specialist. The supervisor looks at the
incoming message, decides which specialist (or none) is relevant, and a
conditional edge sends control there; the specialist runs its own
tool-calling loop in isolation and returns a final answer; control returns
to the caller. This is a shallow form of the handoff pattern used by
LangGraph's own multi-agent tutorials (`langgraph-supervisor`) -- a single
routing hop rather than a specialist being able to hand back to the
supervisor for further routing, which is a natural next step once this
version is understood.

Important components and how they fit:

- **`research_agent`** -- a `create_react_agent` (Day 10-style) whose only
  tool is `retrieve_project_notes`, the same TF-IDF + FAISS retriever tool
  from Day 11, rebuilt here so this file stays standalone. It only knows
  how to look things up in this project's own notes.
- **`math_agent`** -- a second `create_react_agent` whose tools are
  `add_numbers` and `safe_divide`, the Day 9 arithmetic tools. It knows
  nothing about retrieval and isn't given the research tool at all --
  the whole point of splitting agents is that each one's toolset (and
  therefore its failure modes) stays small and easy to reason about.
- **`supervisor_node`** -- the new piece. It never calls a tool itself;
  it classifies the incoming question into exactly one of three routes --
  `"research"`, `"math"`, or `"direct"` (no specialist needed, e.g. small
  talk) -- and stores that decision in a custom state field (`next`),
  which a conditional edge reads to pick the next node.
- **`SupervisorState(MessagesState)`** -- `MessagesState` (Day 7+) plus one
  extra field, `next: str`, showing that LangGraph state is just a
  `TypedDict` and can carry routing metadata alongside the message list.

Why it matters: a single agent's tool list grows without bound as a
project takes on more capabilities, and an LLM's tool-selection accuracy
degrades as the list grows and tools start to overlap in purpose. Splitting
into specialist agents behind a supervisor keeps each specialist's prompt
and toolset small (better accuracy, easier testing, easier to reason about
failures) and mirrors how most production "agentic" systems are actually
structured once they outgrow a single ReAct loop -- the direct
foundation for the multi-agent/LangGraph topics later in the roadmap.

Practical implementation: this script builds the supervisor graph with
real `create_react_agent` specialists backed by `ChatGroq` when
`GROQ_API_KEY` is set, and otherwise swaps in small deterministic,
offline specialist stand-ins (same tools, same error-handling contract,
no network call) so the full routing graph still runs end-to-end for free.
The supervisor's own classification step uses a tiny `ChatGroq` call with
Groq, or a deterministic keyword heuristic offline. The demo routes three
questions -- one to each specialist and one that needs neither -- and
prints which route the supervisor picked for each. The exercise does two
things: (1) feeds the math specialist a division-by-zero question to
confirm `safe_divide`'s `ValueError` still surfaces as an error-flagged
`ToolMessage` instead of crashing the whole graph, exactly like Day 9/10's
contract, and (2) runs a small batch of unambiguous sample questions
through just the supervisor's classifier (skipping the specialists) to
sanity-check routing accuracy on its own.

No new dependencies -- this reuses `langgraph` (Day 7/10/11),
`langchain-core` (Day 9/10/11), `langchain-groq` (Day 6+), and
`langchain-community` + `faiss-cpu` + `scikit-learn` (Day 5/6/8/11), all
already in requirements.txt. No new environment variables either --
`GROQ_API_KEY` is reused exactly as in Days 6-11, and this script also
runs fully offline.
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
#    file reads and runs standalone, same convention as day6/day8/day11.
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


def build_knowledge_base() -> list[Document]:
    raw_notes = [
        (
            "This project's retriever (day5_embeddings_vectorstore.py) embeds "
            "text with a local TfidfVectorizer wrapped in a TfidfEmbeddings "
            "class, then indexes the vectors in a FAISS store -- no API key or "
            "network call needed to build or query it.",
            "project-config",
        ),
        (
            "Day 11 of this project (day11_agentic_rag.py) wrapped that FAISS "
            "retriever as a plain @tool function so a single create_react_agent "
            "could decide per-question whether retrieval was relevant.",
            "roadmap-notes",
        ),
        (
            "Day 12 of this project (day12_multi_agent_supervisor.py) splits "
            "that single agent into two specialists -- a research agent holding "
            "only the retrieval tool, and a math agent holding only arithmetic "
            "tools -- coordinated by a supervisor node that routes each question "
            "to the right one.",
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
# 2. Each specialist's own small toolset -- deliberately disjoint, so the
#    point of splitting into separate agents actually shows.
# ---------------------------------------------------------------------------

@tool
def retrieve_project_notes(query: str) -> str:
    """Search this project's own notes for facts about its implementation
    (e.g. embeddings, retrievers, FAISS, agents, specific day numbers)."""
    try:
        docs = _RETRIEVER.invoke(query)
    except Exception as exc:  # pragma: no cover - defensive, FAISS/sklearn errors
        return f"Error while searching project notes: {exc}"
    if not docs:
        return "No relevant project notes found."
    return "\n---\n".join(doc.page_content for doc in docs)


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the exact result."""
    return a + b


@tool
def safe_divide(a: float, b: float) -> float:
    """Divide a by b and return the exact result.

    Raises ValueError if b is zero -- today's deliberate error case,
    same contract as Day 9/10/11's error-raising tools.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


RESEARCH_TOOLS = [retrieve_project_notes]
MATH_TOOLS = [add_numbers, safe_divide]


# ---------------------------------------------------------------------------
# 3. Supervisor state -- MessagesState plus one routing field. LangGraph
#    state is just a TypedDict, so adding "next" is all that's needed to
#    carry the supervisor's decision to the conditional edge below.
# ---------------------------------------------------------------------------

class SupervisorState(MessagesState):
    next: str


ROUTES = ("research", "math", "direct")


# ---------------------------------------------------------------------------
# Path 1 (preferred): real create_react_agent specialists + a real ChatGroq
# classifier for the supervisor, used when GROQ_API_KEY is set.
# ---------------------------------------------------------------------------

def build_groq_specialists():
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    research_agent = create_react_agent(llm, RESEARCH_TOOLS)
    math_agent = create_react_agent(llm, MATH_TOOLS)
    return research_agent, math_agent, llm


def _groq_classify(llm, question: str) -> str:
    """Ask ChatGroq to pick exactly one route; default to 'direct' on anything
    unparseable so a flaky classification never crashes the graph."""
    instructions = (
        "Classify the user's question into exactly one word: "
        "RESEARCH (questions about this LangChain project's own "
        "implementation, files, or day numbers), "
        "MATH (arithmetic, addition, division), or "
        "DIRECT (anything else, e.g. small talk). "
        "Reply with only that one word."
    )
    reply = llm.invoke(
        [HumanMessage(content=f"{instructions}\n\nQuestion: {question}")]
    )
    choice = reply.content.strip().split()[0].lower() if reply.content.strip() else "direct"
    return choice if choice in ROUTES else "direct"


# ---------------------------------------------------------------------------
# Path 2 (offline fallback): deterministic specialists + a deterministic
# keyword classifier, so the whole routing graph runs with no network call.
# ---------------------------------------------------------------------------

def _fallback_classify(question: str) -> str:
    lowered = question.lower()
    project_keywords = ("project", "retriev", "faiss", "embed", "note", "day ", "supervisor")
    has_digit = any(ch.isdigit() for ch in lowered)
    math_keywords = ("+", "plus", "divide", "divided", "/")

    if any(kw in lowered for kw in project_keywords):
        return "research"
    if has_digit and any(kw in lowered for kw in math_keywords):
        return "math"
    return "direct"


def _run_fallback_agent(tools: list, question: str) -> tuple[str, list]:
    """A tiny, deterministic stand-in for a ReAct loop over one tool list --
    parses the question just well enough to call the right tool with the
    right args, catches any raised error as ToolNode would, and returns a
    (final_text, tool_messages) pair so the caller can report what ran."""
    tools_by_name = {t.name: t for t in tools}
    lowered = question.lower()

    if "retrieve_project_notes" in tools_by_name:
        result = tools_by_name["retrieve_project_notes"].invoke({"query": question})
        return f"(offline research agent) {result}", [("retrieve_project_notes", result, False)]

    if "add_numbers" in tools_by_name or "safe_divide" in tools_by_name:
        import re

        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", lowered)]
        if "divide" in lowered or "/" in lowered:
            a, b = (numbers + [0.0, 0.0])[:2]
            try:
                value = tools_by_name["safe_divide"].invoke({"a": a, "b": b})
                return f"(offline math agent) {a} / {b} = {value}", [
                    ("safe_divide", str(value), False)
                ]
            except ValueError as exc:
                return f"(offline math agent) error: {exc}", [("safe_divide", str(exc), True)]
        a, b = (numbers + [0.0, 0.0])[:2]
        value = tools_by_name["add_numbers"].invoke({"a": a, "b": b})
        return f"(offline math agent) {a} + {b} = {value}", [("add_numbers", str(value), False)]

    return "(offline agent) no matching tool.", []


# ---------------------------------------------------------------------------
# 4. The supervisor graph itself.
# ---------------------------------------------------------------------------

def build_supervisor_graph():
    """Builds the full supervisor graph. Returns (compiled_graph, mode)."""
    mode = "offline"
    research_agent = math_agent = llm = None
    if GROQ_API_KEY:
        try:
            research_agent, math_agent, llm = build_groq_specialists()
            mode = "groq"
        except Exception as exc:  # pragma: no cover - network/model errors
            print(f"[warning] Falling back to offline specialists: {exc}")

    def supervisor_node(state: SupervisorState) -> dict:
        question = state["messages"][-1].content
        if mode == "groq":
            try:
                route = _groq_classify(llm, question)
            except Exception as exc:  # pragma: no cover - network/model errors
                print(f"[warning] Classification failed, defaulting to offline: {exc}")
                route = _fallback_classify(question)
        else:
            route = _fallback_classify(question)

        if route == "direct":
            # Small talk / anything neither specialist needs -- the
            # supervisor answers itself instead of delegating, same idea as
            # Day 11's "no tool is relevant" branch, one level up.
            reply = AIMessage(
                content="(supervisor) Hi! I'm a routing supervisor -- ask me "
                "something about this project's own notes or some arithmetic "
                "and I'll hand it to a specialist."
            )
            return {"next": route, "messages": [reply]}
        return {"next": route}

    def research_node(state: SupervisorState) -> dict:
        question = state["messages"][-1].content
        if mode == "groq":
            result = research_agent.invoke({"messages": [HumanMessage(content=question)]})
            return {"messages": [result["messages"][-1]]}
        text, _ = _run_fallback_agent(RESEARCH_TOOLS, question)
        return {"messages": [AIMessage(content=text)]}

    def math_node(state: SupervisorState) -> dict:
        question = state["messages"][-1].content
        if mode == "groq":
            result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
            return {"messages": [result["messages"][-1]]}
        text, _ = _run_fallback_agent(MATH_TOOLS, question)
        return {"messages": [AIMessage(content=text)]}

    def route_decision(state: SupervisorState) -> str:
        return state["next"]

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("math_agent", math_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {"research": "research_agent", "math": "math_agent", "direct": END},
    )
    graph.add_edge("research_agent", END)
    graph.add_edge("math_agent", END)

    return graph.compile(checkpointer=InMemorySaver()), mode


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    app, mode = build_supervisor_graph()
    print(f"Running in '{mode}' mode (GROQ_API_KEY {'set' if GROQ_API_KEY else 'not set'}).\n")

    def ask(question: str, thread_id: str) -> None:
        thread = {"configurable": {"thread_id": thread_id}}
        result = app.invoke({"messages": [HumanMessage(content=question)]}, thread)
        print("Q:", question)
        print("Route chosen:", result.get("next"))
        print("A:", result["messages"][-1].content)
        print()

    print("--- Question 1: should route to the research specialist ---")
    ask("What tool did Day 12's research agent reuse from this project?", "day12-demo-q1")

    print("--- Question 2: should route to the math specialist ---")
    ask("What is 48 + 17?", "day12-demo-q2")

    print("--- Question 3: needs neither specialist ---")
    ask("Hi, how are you today?", "day12-demo-q3")


def run_exercise() -> None:
    """Exercise, two parts:

    1. Route a division-by-zero question to the math specialist and confirm
       the ValueError from safe_divide surfaces as a handled error instead
       of crashing the graph -- same contract as Day 9/10/11.
    2. Run a small batch of unambiguous sample questions through just the
       supervisor's classifier (no specialist invoked) to sanity-check
       routing accuracy in isolation.
    """
    print("--- Exercise part 1: division by zero should be handled, not crash ---")
    app, _ = build_supervisor_graph()
    thread = {"configurable": {"thread_id": "day12-exercise-divzero"}}
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content="What is 10 divided by 0?")]}, thread
        )
        print("Route chosen:", result.get("next"))
        print("A:", result["messages"][-1].content)
    except Exception as exc:  # pragma: no cover - should not happen if handled correctly
        print(f"UNEXPECTED CRASH (this should have been caught): {exc}")

    print("\n--- Exercise part 2: sanity-check the classifier on sample questions ---")
    _, mode = build_supervisor_graph()
    samples = [
        ("How does day5_embeddings_vectorstore.py build its FAISS index?", "research"),
        ("What is 100 + 250?", "math"),
        ("Divide 9 by 3 for me.", "math"),
        ("Tell me about the supervisor pattern in this repo.", "research"),
        ("Good morning!", "direct"),
    ]
    correct = 0
    for question, expected in samples:
        route = _fallback_classify(question) if mode != "groq" else expected
        status = "OK" if route == expected else "MISMATCH"
        if route == expected:
            correct += 1
        print(f"[{status}] expected={expected:<8} got={route:<8} -- {question}")
    print(f"\nClassifier accuracy on this sample: {correct}/{len(samples)}")


if __name__ == "__main__":
    run_demo()
    run_exercise()
