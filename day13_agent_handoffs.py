"""
Day 13 -- Agent Handoffs: peer-to-peer routing with LangGraph's `Command`
=============================================================================

Day 12 (`day12_multi_agent_supervisor.py`) built a supervisor graph: one
central router node classifies every incoming question and a *static*
conditional edge table (`add_conditional_edges`) sends control to exactly
one specialist, which always returns straight to `END`. That file's own
closing note flagged the obvious next step -- "a natural next step once
this version is understood is letting a specialist hand back to the
supervisor for further routing." Today goes one step further than that:
specialists hand off *directly to each other*, with no trip back through
a central dispatcher in between -- the pattern LangGraph's own prebuilt
multi-agent "swarm" architecture is built on.

What the concept is: a LangGraph node function does not have to just
return a plain state-update dict. It can instead return a
`langgraph.types.Command` object, which bundles a state update
(`update=...`) *and* a routing decision (`goto=...`) into a single return
value. Day 12 kept those two concerns separate -- a node returned a dict,
and a *different* piece of code (a conditional edge function reading
`state["next"]`) decided where to go next. `Command` lets the node that
just finished reasoning about a question decide, in the same breath,
both what happened and what should happen next -- which is exactly what
"hand this off to the other specialist" needs: the research agent
finishing one sub-question doesn't need a supervisor to tell it the
remaining sub-question is a math question; it can route there itself.

Important components and how they work:

- **`langgraph.types.Command`** -- the new piece today. `Command(update={
  "messages": [...], "visited": [...]}, goto="math_agent")` both appends
  to shared state *and* transfers control to the `"math_agent"` node in
  one step. A node that returns a `Command` needs no outgoing
  `add_edge`/`add_conditional_edges` call at all -- the routing lives in
  the node's own return value instead of in separate graph wiring.
- **A compound-question dispatcher** (`dispatch_node`) -- classifies a
  question into "needs research", "needs math", both, or neither, and
  issues the *first* `Command(goto=...)` -- this one hop still looks like
  Day 12's supervisor, but it only ever fires once per conversation turn.
- **Peer handoffs** (`research_node`, `math_node`) -- after answering its
  own half of a compound question, each specialist checks whether the
  *other* half is still unanswered and, if so, returns
  `Command(goto=<the other specialist>, ...)` directly -- a hop Day 12's
  architecture had no way to express, since every one of its edges
  pointed at `END`.
- **A cycle guard** (`state["visited"]`) -- today's deliberate edge case.
  Without it, two specialists that each think the other should go next
  could hand off forever. Each node appends its own name to
  `state["visited"]` before handing off, and refuses to hand off to a
  specialist whose name is already in that list -- ending the turn there
  instead of looping.

Why it matters: a central supervisor is simple but becomes a bottleneck
once a question genuinely needs several specialists in sequence --
every single hop pays the round trip back through the dispatcher. Direct
handoffs let specialists compose themselves into a short pipeline for
one compound question, which is the actual mechanism behind LangGraph's
`langgraph-swarm` style multi-agent systems, and a direct stepping stone
toward the advanced multi-agent/LangGraph topics later in the roadmap.

Practical implementation: `day13_agent_handoffs.py` reuses Day 9/11/12's
tools (`retrieve_project_notes`, `add_numbers`, `safe_divide`) behind two
specialists, backed by real `create_react_agent` + `ChatGroq` when
`GROQ_API_KEY` is set, and otherwise by the same deterministic offline
stand-ins Day 12 used, so the whole handoff graph runs end-to-end with no
network call. The demo runs three questions: one that only needs
research, one that only needs math, and one *compound* question needing
both -- printing `state["visited"]` each time to show the actual path
taken through the graph (one hop for the simple questions, two for the
compound one, with no supervisor revisited in between). The exercise
re-divides by zero (Day 9/10/11/12's recurring error case) *inside* a
compound handoff to confirm a specialist's own error handling still
works mid-handoff, then deliberately feeds a question that would make
both specialists want to hand off to each other forever, to confirm the
cycle guard stops it after one hop instead of looping.

No new dependencies -- this reuses `langgraph` (Day 7/10/11/12),
`langchain-core` (Day 9/10/11/12), `langchain-groq` (Day 6+), and
`langchain-community` + `faiss-cpu` + `scikit-learn` (Day 5/6/8/11/12),
all already in requirements.txt. No new environment variables either --
`GROQ_API_KEY` is reused exactly as in Days 6-12, and the full offline
run was verified end-to-end today.
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
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.types import Command
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "This script needs 'langgraph'. Install with: pip install langgraph"
    ) from exc

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# 1. Day 5's offline embeddings backend, repeated here (not imported) so this
#    file reads and runs standalone, same convention as day6/day8/day11/day12.
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
            "Day 12 of this project (day12_multi_agent_supervisor.py) built a "
            "supervisor graph that routes each question to one of two "
            "specialist agents and always returns straight to END afterward.",
            "roadmap-notes",
        ),
        (
            "Day 13 of this project (day13_agent_handoffs.py) replaces that "
            "static supervisor edge table with LangGraph's Command object, so "
            "a specialist can hand a compound question directly to the other "
            "specialist without going back through a central router.",
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
# 2. Each specialist's own small toolset -- same split as Day 12, deliberately
#    disjoint so a compound question genuinely needs both specialists.
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

    Raises ValueError if b is zero -- today's recurring deliberate error
    case, same contract as Day 9/10/11/12's error-raising tools.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


RESEARCH_TOOLS = [retrieve_project_notes]
MATH_TOOLS = [add_numbers, safe_divide]


# ---------------------------------------------------------------------------
# 3. Handoff state -- MessagesState plus a `visited` list. This is the cycle
#    guard: every node appends its own name before handing off, and refuses
#    to hand off to a name that's already there.
# ---------------------------------------------------------------------------

class HandoffState(MessagesState):
    visited: list[str]


def _needs_research(question: str) -> bool:
    lowered = question.lower()
    return any(
        kw in lowered
        for kw in ("project", "retriev", "faiss", "embed", "note", "day ", "supervisor", "handoff")
    )


def _needs_math(question: str) -> bool:
    lowered = question.lower()
    has_digit = any(ch.isdigit() for ch in lowered)
    return has_digit and any(kw in lowered for kw in ("+", "plus", "divide", "divided", "/"))


def _parse_math(question: str) -> tuple[float, float, bool]:
    """Pull the first two numbers out of a question and say whether it's a
    division. Shared by the Groq and offline math paths."""
    import re

    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", question)]
    a, b = (numbers + [0.0, 0.0])[:2]
    is_division = "divide" in question.lower() or "/" in question
    return a, b, is_division


# ---------------------------------------------------------------------------
# Path 1 (preferred): real create_react_agent specialists, used when
# GROQ_API_KEY is set. Path 2 (offline fallback): deterministic stand-ins,
# same contract, no network call -- exactly Day 12's two-path shape.
# ---------------------------------------------------------------------------

def build_groq_specialists():
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    research_agent = create_react_agent(llm, RESEARCH_TOOLS)
    math_agent = create_react_agent(llm, MATH_TOOLS)
    return research_agent, math_agent


def _run_research(question: str, mode: str, research_agent) -> str:
    if mode == "groq":
        result = research_agent.invoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content
    return f"(offline research agent) {retrieve_project_notes.invoke({'query': question})}"


def _run_math(question: str, mode: str, math_agent) -> str:
    a, b, is_division = _parse_math(question)
    if mode == "groq":
        result = math_agent.invoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content
    if is_division:
        try:
            value = safe_divide.invoke({"a": a, "b": b})
            return f"(offline math agent) {a} / {b} = {value}"
        except ValueError as exc:
            # Basic error handling: a tool error mid-handoff is reported back
            # as text, not raised -- so the graph still finishes the turn
            # instead of crashing, same contract as Day 9/10/11/12.
            return f"(offline math agent) error: {exc}"
    value = add_numbers.invoke({"a": a, "b": b})
    return f"(offline math agent) {a} + {b} = {value}"


# ---------------------------------------------------------------------------
# 4. The handoff graph itself. Every node after `dispatch` returns a
#    `Command` -- state update and routing decision in one return value --
#    instead of a plain dict plus a separately-wired conditional edge.
# ---------------------------------------------------------------------------

def build_handoff_graph():
    """Builds the full handoff graph. Returns (compiled_graph, mode)."""
    mode = "offline"
    research_agent = math_agent = None
    if GROQ_API_KEY:
        try:
            research_agent, math_agent = build_groq_specialists()
            mode = "groq"
        except Exception as exc:  # pragma: no cover - network/model errors
            print(f"[warning] Falling back to offline specialists: {exc}")

    def original_question(state: HandoffState) -> str:
        # The first message in the thread is always the user's original
        # (possibly compound) question -- later messages are specialists'
        # own answers, which is why handoff decisions re-read message[0]
        # rather than the most recent message.
        return state["messages"][0].content

    def dispatch_node(state: HandoffState) -> Command:
        question = original_question(state)
        needs_research = _needs_research(question)
        needs_math = _needs_math(question)
        if needs_research:
            return Command(goto="research_agent", update={"visited": []})
        if needs_math:
            return Command(goto="math_agent", update={"visited": []})
        # Neither specialist is relevant -- answer directly, same idea as
        # Day 11's "no tool is relevant" branch and Day 12's "direct" route.
        reply = AIMessage(
            content="(dispatcher) Ask me something about this project's own "
            "notes, some arithmetic, or both in one question."
        )
        return Command(goto="__end__", update={"messages": [reply], "visited": []})

    def research_node(state: HandoffState) -> Command:
        question = original_question(state)
        answer = _run_research(question, mode, research_agent)
        visited = state.get("visited", []) + ["research_agent"]
        update = {"messages": [AIMessage(content=answer)], "visited": visited}

        if _needs_math(question) and "math_agent" not in visited:
            # Direct peer handoff -- no trip back through a dispatcher node.
            return Command(goto="math_agent", update=update)
        return Command(goto="__end__", update=update)

    def math_node(state: HandoffState) -> Command:
        question = original_question(state)
        answer = _run_math(question, mode, math_agent)
        visited = state.get("visited", []) + ["math_agent"]
        update = {"messages": [AIMessage(content=answer)], "visited": visited}

        if _needs_research(question) and "research_agent" not in visited:
            return Command(goto="research_agent", update=update)
        return Command(goto="__end__", update=update)

    graph = StateGraph(HandoffState)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("math_agent", math_node)
    graph.set_entry_point("dispatch")
    # No add_edge / add_conditional_edges calls for routing between
    # dispatch/research_agent/math_agent -- every hop between them is
    # decided by the Command each node already returned.

    return graph.compile(checkpointer=InMemorySaver()), mode


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    app, mode = build_handoff_graph()
    print(f"Running in '{mode}' mode (GROQ_API_KEY {'set' if GROQ_API_KEY else 'not set'}).\n")

    def ask(question: str, thread_id: str) -> None:
        thread = {"configurable": {"thread_id": thread_id}}
        result = app.invoke({"messages": [HumanMessage(content=question)]}, thread)
        print("Q:", question)
        print("Path taken:", result.get("visited", []))
        print("Final answer:", result["messages"][-1].content)
        print()

    print("--- Question 1: only the research specialist should run ---")
    ask("What tool does Day 13's dispatcher replace from Day 12?", "day13-demo-q1")

    print("--- Question 2: only the math specialist should run ---")
    ask("What is 48 + 17?", "day13-demo-q2")

    print("--- Question 3: compound -- both specialists, one direct handoff ---")
    ask(
        "What does this project's FAISS retriever use for embeddings, and "
        "also, what is 100 + 250?",
        "day13-demo-q3",
    )


def run_exercise() -> None:
    """Exercise, two parts:

    1. A compound question whose math half divides by zero -- confirms the
       math specialist's own error handling still reports the error as text
       (not a crash) even when it was reached via a mid-turn handoff rather
       than as the first node, same contract as Day 9/10/11/12.
    2. A question phrased so that both specialists' keyword heuristics think
       the *other* one should run next -- confirms the `visited` cycle guard
       stops the handoff after one hop each way instead of looping forever.
    """
    print("--- Exercise part 1: division by zero inside a handoff, not a crash ---")
    app, _ = build_handoff_graph()
    thread = {"configurable": {"thread_id": "day13-exercise-divzero"}}
    try:
        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Tell me about this project's retriever, and "
                        "what is 10 divided by 0?"
                    )
                ]
            },
            thread,
        )
        print("Path taken:", result.get("visited", []))
        print("Final answer:", result["messages"][-1].content)
    except Exception as exc:  # pragma: no cover - should not happen if handled correctly
        print(f"UNEXPECTED CRASH (this should have been caught): {exc}")

    print("\n--- Exercise part 2: cycle guard stops a would-be infinite handoff ---")
    app, _ = build_handoff_graph()
    thread = {"configurable": {"thread_id": "day13-exercise-cycle"}}
    # Deliberately mentions both research and math keywords so each
    # specialist's own heuristic would ask to hand off to the other again --
    # without `visited`, research_agent -> math_agent -> research_agent ->
    # ... would never stop.
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="project notes day retriever faiss embed plus "
                    "divide 9 / 3"
                )
            ]
        },
        thread,
    )
    print("Path taken:", result.get("visited", []))
    print(
        "Hops recorded:",
        len(result.get("visited", [])),
        "(cycle guard stopped it instead of looping forever)",
    )
    print("Final answer:", result["messages"][-1].content)


if __name__ == "__main__":
    run_demo()
    run_exercise()
