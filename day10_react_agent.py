"""
Day 10: Agents - from a hand-rolled tool loop (Day 9) to a graph-based ReAct agent

Day 9 (day9_tool_calling.py) wrote the tool-calling loop by hand:

    invoke -> if tool_calls: run them, append ToolMessages, invoke again -> else: done

That loop IS a ReAct agent ("Reason + Act"): the model reasons about what to do,
acts by calling a tool, observes the result, and repeats until it has enough
information to answer. Today wires that same loop up as an explicit LangGraph
*graph* instead of a plain while-loop, which is what LangGraph's own
`langgraph.prebuilt.create_react_agent` builds internally:

    agent_node --(tool_calls present)--> tools_node --> agent_node --(loop)-->
    agent_node --(no tool_calls)--> END

Key pieces:

- `create_react_agent(model, tools, checkpointer=...)` - a prebuilt StateGraph
  with exactly the agent/tools loop above already wired up, plus message-list
  state and (when given a checkpointer) multi-turn memory via `thread_id`,
  exactly like Day 7/8's graphs.
- `langgraph.prebuilt.ToolNode` - a ready-made graph node that reads the last
  message's `tool_calls`, runs each matching tool, and returns the resulting
  `ToolMessage`s. It is what `create_react_agent` uses internally for the
  "Act" half of the loop, and it is reused directly below in the offline
  fallback graph so both code paths share the exact same tool-execution step.
- Conditional routing - `tools_condition` (or the hand-written equivalent
  below) inspects the agent's latest `AIMessage` and decides whether to loop
  back into the tools node or stop, which is the one piece Day 9's plain
  while-loop had to spell out by hand and a graph expresses as an edge.

Why it matters: Day 9 showed what an agent loop actually does by writing it
manually once. Today shows the same loop as the reusable, inspectable graph
structure that every LangGraph agent (and everything from Day 31 onward on
the roadmap - multi-agent systems, LangGraph deployments) is built from.

Practical implementation: this script defines three new tools (`word_count`,
`is_prime`, `lookup_day_topic` - a callable stand-in for "ask the journal"),
builds a real `create_react_agent` when `GROQ_API_KEY` is set, and otherwise
builds an equivalent hand-wired `StateGraph` (same tools, same `ToolNode`,
a deterministic keyword-router standing in for the LLM "agent" node, and a
conditional edge) so the full graph runs end-to-end with no network call.
The demo asks one question that needs two different tool calls in a single
turn, and reuses a `thread_id` to show the follow-up turn still resolves
correctly against the running history. The exercise deliberately asks about
a prime check
on a bad input (`is_prime(1)`), which raises `ValueError`, and confirms the
graph's tool node reports the error back as a `ToolMessage` instead of
crashing - the same error-handling contract `safe_divide` demonstrated in
Day 9, now inside a graph instead of a manual `try/except`.

No new dependencies: `langgraph.prebuilt` ships with `langgraph`, already in
requirements.txt (Day 7). No new environment variables either - GROQ_API_KEY
is reused exactly as in Day 6-9, and this script also runs fully offline.
"""

import os

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Tools - plain, typed, documented Python functions, same contract as Day 9
# ---------------------------------------------------------------------------

@tool
def word_count(text: str) -> int:
    """Count how many whitespace-separated words are in a piece of text."""
    return len(text.split())


@tool
def is_prime(n: int) -> bool:
    """Return True if the integer n is prime, else False.

    Raises ValueError for n < 2, since primality is undefined there - this
    is the deliberate error case used in today's exercise.
    """
    if n < 2:
        raise ValueError(f"Primality is only defined for integers >= 2, got {n}.")
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for divisor in range(3, int(n ** 0.5) + 1, 2):
        if n % divisor == 0:
            return False
    return True


# A tiny stand-in for "ask the journal" - a real version would wrap Day 6's
# retriever (search_project_notes in day9_tool_calling.py already does
# exactly that for free-text notes); this one answers a structured lookup.
_DAY_TOPICS = {
    1: "LangChain fundamentals and first setup",
    2: "LCEL runnables: RunnableParallel, RunnableLambda, RunnablePassthrough",
    3: "Document loaders: TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader",
    4: "Text splitting: CharacterTextSplitter, RecursiveCharacterTextSplitter",
    5: "Embeddings and vector stores: Embeddings interface, FAISS",
    6: "Retrieval-Augmented Generation: wiring a retriever into an LCEL chain",
    7: "Conversation memory: LangGraph StateGraph + InMemorySaver checkpointer",
    8: "Conversational RAG: history-aware query rewriting before retrieval",
    9: "Tool calling: binding Python functions to an LLM with bind_tools",
}


@tool
def lookup_day_topic(day_number: int) -> str:
    """Look up which LangChain topic was studied on a given day number of this journey."""
    return _DAY_TOPICS.get(
        day_number, f"No journal record for day {day_number} yet."
    )


TOOLS = [word_count, is_prime, lookup_day_topic]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Path 1 (preferred): a real create_react_agent, used when GROQ_API_KEY is set
# ---------------------------------------------------------------------------

def build_groq_agent():
    """Build a real LangGraph ReAct agent backed by ChatGroq."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    checkpointer = InMemorySaver()
    # create_react_agent wires up exactly the agent/tools loop described in
    # the module docstring: it returns a compiled graph with a MessagesState,
    # identical in shape to the hand-wired one built below.
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(llm, TOOLS, checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Path 2 (offline fallback): the same graph, hand-wired with a deterministic
# keyword-router standing in for the LLM's reasoning step.
# ---------------------------------------------------------------------------

def _fake_llm_turn(messages: list) -> AIMessage:
    """Deterministic stand-in for an LLM's reasoning step.

    Looks at the latest HumanMessage and decides which tool(s) to call, in
    the same shape a real tool-calling LLM would return: an AIMessage whose
    `tool_calls` list names a tool and its arguments. If every tool this
    question needed has already been answered (there are ToolMessages for
    all of them), it synthesizes a final text answer instead - mirroring
    the "no tool_calls left -> done" branch of the real loop.
    """
    # Find the most recent HumanMessage and reason only over what happened
    # *since* it - this is a multi-turn checkpointer (Day 7 style), so
    # earlier turns' ToolMessages are still in `messages` but must not be
    # mistaken for work already done on the current question.
    last_human_index = 0
    question = ""
    for idx, msg in enumerate(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_index = idx
            question = msg.content

    current_turn = messages[last_human_index:]

    already_called = {
        msg.name
        for msg in current_turn
        if msg.__class__.__name__ == "ToolMessage"
    }

    pending_calls = []

    if "word" in question.lower() and "word_count" not in already_called:
        # Pull the quoted phrase out of the question, e.g. 'count of "a b c"'.
        if '"' in question:
            phrase = question.split('"')[1]
        else:
            phrase = question
        pending_calls.append(
            {"name": "word_count", "args": {"text": phrase}, "id": "call_word_count"}
        )

    if "prime" in question.lower() and "is_prime" not in already_called:
        digits = "".join(ch if ch.isdigit() else " " for ch in question).split()
        n = int(digits[0]) if digits else 0
        pending_calls.append(
            {"name": "is_prime", "args": {"n": n}, "id": "call_is_prime"}
        )

    if "day" in question.lower() and "lookup_day_topic" not in already_called:
        digits = "".join(ch if ch.isdigit() else " " for ch in question).split()
        # Heuristic: the day number is whichever digit group isn't the one
        # already consumed as the prime-check argument.
        day_candidates = digits[1:] if "prime" in question.lower() else digits
        if day_candidates:
            pending_calls.append(
                {
                    "name": "lookup_day_topic",
                    "args": {"day_number": int(day_candidates[0])},
                    "id": "call_lookup_day_topic",
                }
            )

    if pending_calls:
        return AIMessage(content="", tool_calls=pending_calls)

    # No more tools needed: summarize this turn's ToolMessages only.
    observations = [
        f"{msg.name}={msg.content}"
        for msg in current_turn
        if msg.__class__.__name__ == "ToolMessage"
    ]
    summary = "; ".join(observations) if observations else "I don't have an answer."
    return AIMessage(content=f"(offline fallback) {summary}")


def _agent_node(state: MessagesState) -> dict:
    reply = _fake_llm_turn(state["messages"])
    return {"messages": [reply]}


def _should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_fallback_agent():
    """Hand-wire the same agent/tools loop create_react_agent builds internally."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    # ToolNode is the same ready-made node the real create_react_agent path
    # uses: it reads tool_calls off the last message, runs them, and turns
    # any exception into a ToolMessage instead of raising - exactly the
    # error-handling behaviour today's exercise relies on.
    # handle_tool_errors=True turns a raised exception (e.g. is_prime(1))
    # into an error-flagged ToolMessage instead of propagating and crashing
    # the graph - the graph-level equivalent of Day 9's manual try/except.
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

    thread = {"configurable": {"thread_id": "day10-demo"}}

    print("--- Turn 1: a question needing two tool calls ---")
    question_1 = 'How many words are in "LangChain agents are fun to build" and what topic was day 7?'
    result = agent.invoke({"messages": [HumanMessage(content=question_1)]}, thread)
    print("Q:", question_1)
    print("A:", result["messages"][-1].content)

    print("\n--- Turn 2: reuses the same thread_id, follow-up memory check ---")
    question_2 = "And what was day 9 about?"
    result = agent.invoke({"messages": [HumanMessage(content=question_2)]}, thread)
    print("Q:", question_2)
    print("A:", result["messages"][-1].content)
    print(
        f"(message history length on this thread: {len(result['messages'])} messages)"
    )


def run_exercise() -> None:
    """Exercise: force the prime-check tool into its documented error case."""
    print("\n--- Exercise: is_prime(1) should raise, not crash the graph ---")
    agent, _ = build_agent()
    thread = {"configurable": {"thread_id": "day10-exercise"}}
    bad_question = "Is 1 a prime number?"
    result = agent.invoke({"messages": [HumanMessage(content=bad_question)]}, thread)

    tool_messages = [
        msg for msg in result["messages"] if msg.__class__.__name__ == "ToolMessage"
    ]
    for msg in tool_messages:
        status = "ERROR" if getattr(msg, "status", None) == "error" else "ok"
        print(f"tool={msg.name} status={status} content={msg.content}")

    print("Final answer:", result["messages"][-1].content)


if __name__ == "__main__":
    run_demo()
    run_exercise()
