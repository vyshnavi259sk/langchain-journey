"""
Day 14: Human-in-the-Loop Agents — LangGraph's interrupt() for approval workflows
===================================================================================

Day 9-13 all built agents that act immediately: the moment the agent (or its
deterministic offline stand-in) decides to call a tool, that tool runs, no
questions asked. That's fine for a calculator or a read-only retriever, but
real agents are often handed *sensitive* tools too -- "send this email",
"charge this card", "delete this record" -- where you want a human to approve
the specific call before it executes, not just trust the model's judgement.

Today's topic is LangGraph's **human-in-the-loop** primitive: `interrupt()`.
Calling `interrupt(payload)` inside a node pauses the graph entirely --
execution stops, the payload surfaces to whoever is driving the graph, and
the graph's state is durably checkpointed exactly where it paused. Nothing
resumes until the caller later invokes the graph again with
`Command(resume=<value>)` on the *same thread_id*. From the node's point of
view, `interrupt()` just returns that resume value, as if it were a normal
function call that happened to take a very long coffee break.

Important components and how they work:

- **`interrupt(payload)`** (`langgraph.types.interrupt`) -- pauses the graph.
  `payload` is anything JSON-ish describing what needs a decision; it is
  surfaced to the caller via `result["__interrupt__"]`. The *second* time this
  node runs (after a resume), `interrupt()` doesn't pause again -- it returns
  the value passed to `Command(resume=...)` instead. This only works because
  of a checkpointer: without one, there is nothing to resume *from*.
- **`Command(resume=value)`** -- instead of passing a normal input dict to
  `graph.invoke()`, passing a `Command` tells LangGraph "continue the paused
  thread with this value," re-entering the interrupted node right where it
  left off.
- **`MemorySaver`** (Day 7's checkpointer, reused unchanged) -- the mechanism
  that makes "pause now, resume later, possibly in a different process" work
  at all. A real deployment would swap this for a durable backend (Postgres,
  SQLite); the interrupt/resume contract is identical either way.
- **A conditional edge gated on tool sensitivity** -- `SENSITIVE_TOOLS` marks
  which tool names require approval. Everything else (Day 9-13's
  `add_numbers`, `safe_divide`-style tools) still runs immediately, so
  human-in-the-loop is opt-in per tool, not a blanket slowdown.

Why it matters: this is the difference between an agent you can hand
read-only or reversible tools and one you can trust with anything that costs
money, sends real messages, or deletes data. Instead of either (a) never
letting agents touch sensitive actions, or (b) trusting them blindly,
`interrupt()` lets the *specific, concrete call* ("send_notification to
finance-team: 'Invoice #4412 is overdue'") be shown to a human before it
executes -- while the rest of the agent's reasoning still runs autonomously.

Practical implementation completed today: `day14_human_in_the_loop.py` builds
a small LangGraph agent with two tools -- `add_numbers` (safe, Day 9-style)
and `send_notification` (marked sensitive). The agent node is a deterministic
keyword router when `GROQ_API_KEY` is unset (same offline-first pattern as
every agent day since Day 9/10), or a real `ChatGroq` bound to both tools
otherwise. A `human_approval` node calls `interrupt()` only when the agent
wants to call a sensitive tool; approving resumes into the real tool call,
rejecting injects a `ToolMessage` saying the action was declined and skips
execution entirely -- the agent never silently retries a rejected action.
The demo runs three scenarios on three separate `thread_id`s: a safe-tool
call that never pauses, a sensitive call that gets approved, and a sensitive
call that gets rejected. The exercise simulates an approver who rejects the
first sensitive request but approves a second, slightly different one on the
same thread, confirming each pending interrupt is independent and a
rejection doesn't poison later calls on that thread.

No new environment variables -- `GROQ_API_KEY` is reused exactly as in Days
6-13 (this file runs fully offline without it). New dependency: none beyond
`langgraph` (already in requirements.txt since Day 7); `interrupt`/`Command`
ship with the same package.
"""

import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Tools: one safe, one sensitive (requires human approval before it runs)
# ---------------------------------------------------------------------------
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together. Safe -- runs immediately, no approval needed."""
    return a + b


@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification message to a recipient.

    SENSITIVE: this has a real-world side effect (in a production system it
    would hit an email/Slack/SMS API), so it requires human approval before
    it actually executes.
    """
    # In a real system this would call an email/Slack API. Here it's a mock
    # so the demo has no external dependency, but the approval gate around
    # it is the real point of today's lesson.
    return f"[SENT] to {recipient}: '{message}'"


ALL_TOOLS = [add_numbers, send_notification]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
SENSITIVE_TOOLS = {"send_notification"}

# ---------------------------------------------------------------------------
# Optional real LLM, offline fallback otherwise (same pattern as Day 9-13)
# ---------------------------------------------------------------------------
if GROQ_API_KEY:
    from langchain_groq import ChatGroq

    _llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    _llm_with_tools = _llm.bind_tools(ALL_TOOLS)
else:
    _llm_with_tools = None


def _offline_router(last_human_text: str) -> AIMessage:
    """Deterministic stand-in for an LLM's tool-choice reasoning.

    Recognizes two patterns well enough to demo both the safe and the
    sensitive path without ever making a network call.
    """
    text = last_human_text.lower()

    # "notify <recipient>: <message>" / "send <recipient> a message: <message>"
    notify_match = re.search(
        r"notify ([\w.@-]+)(?: that| with| saying)?:?\s*(.+)", text
    )
    if notify_match or "send a notification" in text or "send notification" in text:
        if notify_match:
            recipient, message = notify_match.group(1), notify_match.group(2)
        else:
            recipient, message = "finance-team", last_human_text
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "send_notification",
                    "args": {"recipient": recipient.strip(), "message": message.strip()},
                    "id": "call_notify_1",
                }
            ],
        )

    # "add N and M" / "what is N + M"
    add_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:\+|and|plus)\s*(-?\d+(?:\.\d+)?)", text)
    if add_match:
        a, b = float(add_match.group(1)), float(add_match.group(2))
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "add_numbers", "args": {"a": a, "b": b}, "id": "call_add_1"}
            ],
        )

    return AIMessage(content="I don't have a tool for that request.")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def agent_node(state: MessagesState) -> dict:
    """Decide what to do next: call a tool, or answer directly.

    Offline mode has to be careful here: `_offline_router` only recognizes
    a request's *initial* phrasing. If this node reran the same regex after
    a tool (or a rejection) already produced a `ToolMessage`, it would just
    re-issue the same tool call forever. So once the most recent message is
    a `ToolMessage`, the offline path always finalizes with a plain-content
    answer instead of routing again -- a real LLM would do this naturally by
    reading the tool result itself, which is exactly what it's given here.
    """
    if _llm_with_tools is not None:
        response = _llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        response = AIMessage(content=f"Done -- {last_message.content}")
    else:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        response = _offline_router(last_human)
    return {"messages": [response]}


def human_approval_node(state: MessagesState) -> Command[Literal["tools", "agent"]]:
    """Pause the graph for a sensitive tool call; resume routes on the verdict.

    Only ever reached when the conditional edge below has already confirmed
    the pending call targets a sensitive tool, so this node doesn't need to
    re-check that itself.
    """
    last_message = state["messages"][-1]
    pending_call = last_message.tool_calls[0]

    # This is where execution actually pauses. The dict passed in becomes
    # `result["__interrupt__"][0].value` for whoever is driving the graph.
    # On resume, `interrupt(...)` returns the value from Command(resume=...)
    # instead of pausing again -- same call site, second time through.
    decision = interrupt(
        {
            "action": "approve_tool_call",
            "tool": pending_call["name"],
            "args": pending_call["args"],
            "question": (
                f"Approve calling `{pending_call['name']}` with "
                f"{pending_call['args']}? (approve / reject)"
            ),
        }
    )

    if decision == "approve":
        return Command(goto="tools")

    # Rejected: inject a ToolMessage so the agent sees *why* nothing ran,
    # and route back to the agent instead of executing the sensitive tool.
    rejection = ToolMessage(
        content=f"Human reviewer rejected this call to `{pending_call['name']}`.",
        tool_call_id=pending_call["id"],
    )
    return Command(goto="agent", update={"messages": [rejection]})


def route_after_agent(state: MessagesState) -> str:
    """After the agent node: no tool call -> end; sensitive -> approval; else -> tools."""
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    if last_message.tool_calls[0]["name"] in SENSITIVE_TOOLS:
        return "human_approval"
    return "tools"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"human_approval": "human_approval", "tools": "tools", END: END},
    )
    # human_approval returns a Command with its own `goto`, so no static
    # edge is declared from it -- the node itself decides "tools" or "agent".
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def run_to_completion_or_pause(app, inputs_or_command, config):
    """Invoke the graph and report whether it finished or paused on a question."""
    result = app.invoke(inputs_or_command, config=config)
    if "__interrupt__" in result:
        pending = result["__interrupt__"][0].value
        print(f"  [PAUSED] {pending['question']}")
        return result, pending
    final_message = result["messages"][-1]
    print(f"  [DONE]   {final_message.content}")
    return result, None


def demo():
    app = build_graph()
    print(f"Mode: {'ChatGroq (live API)' if _llm_with_tools else 'offline deterministic router'}\n")

    # --- Scenario 1: safe tool, never pauses -------------------------------
    print("Scenario 1: safe tool call (add_numbers) -- should run with no pause")
    config_1 = {"configurable": {"thread_id": "demo-thread-1"}}
    run_to_completion_or_pause(
        app, {"messages": [HumanMessage(content="what is 12 and 30")]}, config_1
    )

    # --- Scenario 2: sensitive tool call, approved -------------------------
    print("\nScenario 2: sensitive tool call (send_notification) -- approved")
    config_2 = {"configurable": {"thread_id": "demo-thread-2"}}
    _, pending = run_to_completion_or_pause(
        app,
        {"messages": [HumanMessage(content="notify finance-team: invoice 4412 is overdue")]},
        config_2,
    )
    assert pending is not None, "expected scenario 2 to pause for approval"
    print("  -> approver says: approve")
    run_to_completion_or_pause(app, Command(resume="approve"), config_2)

    # --- Scenario 3: sensitive tool call, rejected -------------------------
    print("\nScenario 3: sensitive tool call (send_notification) -- rejected")
    config_3 = {"configurable": {"thread_id": "demo-thread-3"}}
    _, pending = run_to_completion_or_pause(
        app,
        {"messages": [HumanMessage(content="notify oncall: database is down")]},
        config_3,
    )
    assert pending is not None, "expected scenario 3 to pause for approval"
    print("  -> approver says: reject")
    run_to_completion_or_pause(app, Command(resume="reject"), config_3)

    # -----------------------------------------------------------------------
    # Exercise: an approver rejects one request, then approves a second,
    # different request on the *same thread* -- confirming that a rejection
    # doesn't poison the thread, and each interrupt is handled independently.
    # -----------------------------------------------------------------------
    print("\nExercise: reject once, then approve a different request on the same thread")
    config_ex = {"configurable": {"thread_id": "demo-thread-exercise"}}

    _, pending = run_to_completion_or_pause(
        app,
        {"messages": [HumanMessage(content="notify security: unrecognized login attempt")]},
        config_ex,
    )
    assert pending is not None
    assert pending["tool"] == "send_notification"
    print("  -> approver says: reject (first request)")
    state_after_reject, pending_after_reject = run_to_completion_or_pause(
        app, Command(resume="reject"), config_ex
    )
    # After a rejection the agent should see the ToolMessage and stop asking
    # the sensitive tool again on its own -- it should answer directly here.
    assert pending_after_reject is None, "agent should not re-attempt a rejected call unprompted"

    _, pending_2 = run_to_completion_or_pause(
        app,
        {"messages": [HumanMessage(content="notify security: login attempt was verified as me")]},
        config_ex,
    )
    assert pending_2 is not None, "a second, distinct request should still pause independently"
    print("  -> approver says: approve (second, distinct request)")
    final_state, pending_after_approve = run_to_completion_or_pause(
        app, Command(resume="approve"), config_ex
    )
    assert pending_after_approve is None
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert any("[SENT]" in m.content for m in tool_messages), (
        "expected the approved call's ToolMessage to show it actually executed"
    )
    print("  Exercise passed: rejection and approval on the same thread stayed independent.")


if __name__ == "__main__":
    demo()
