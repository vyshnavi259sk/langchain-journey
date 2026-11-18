"""
Day 7 — Conversation Memory and Chat History with LangGraph Persistence
=========================================================================

Day 6 built a RAG chain that answers a *single* question well, but it has
no memory: ask a follow-up like "what about that?" and the chain has no
idea what "that" refers to, because each `.invoke()` call is completely
independent of the last.

Today's topic is **conversation memory** — giving a chain access to prior
turns so follow-up questions work. The first version of this script used
`RunnableWithMessageHistory` + `InMemoryChatMessageHistory`
(`langchain_core.runnables.history` / `langchain_core.chat_history`), which
is the pattern older LangChain tutorials still show — but running it here
raises:

    LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated.
    Use LangGraph's built-in persistence instead.

So this implementation uses the current recommended replacement:
**LangGraph's graph + checkpointer** pattern.

Key pieces introduced:

- **`langgraph.graph.StateGraph` / `MessagesState`** — a graph of one or
  more nodes that share a state object. `MessagesState` is a ready-made
  state shape with a single `messages` key whose updates are *merged*
  (appended) rather than overwritten, via LangGraph's `add_messages`
  reducer — so returning `{"messages": [new_message]}` from a node adds
  to the conversation instead of replacing it.
- **A checkpointer** (`langgraph.checkpoint.memory.InMemorySaver`) — saves
  the graph's state after every step, keyed by a `thread_id`. Passing the
  same `thread_id` on a later `.invoke()` call resumes that conversation
  automatically; a new `thread_id` starts a fresh, isolated one. This is
  the direct replacement for `RunnableWithMessageHistory`'s `session_id`.
- **`trim_messages`** (`langchain_core.messages`, still a current,
  non-deprecated API) — trims the message list to a budget before it
  reaches the model, since stored history grows every turn but an LLM's
  context window (and a pay-per-token bill) does not.

Why it matters: almost every real LLM product (chatbots, coding
assistants, support bots) needs multi-turn memory, and LangGraph's
checkpointer is the officially recommended way to add it as of current
LangChain versions — it is also the foundation the roadmap's later agent
and multi-agent topics (Day 31+) build on, so this is a deliberate early
look at LangGraph before using it for full agents.

New dependency: `langgraph` (added to `requirements.txt`). No new
environment variables were needed — this still only reads the existing
`GROQ_API_KEY`, and runs a deterministic offline stand-in with no network
call at all when that key is absent, so it is fully runnable either way.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = SystemMessage(content="You are a concise, helpful assistant. Keep answers short.")


# ---------------------------------------------------------------------------
# 1. The "respond" step: a real LLM call when a key is configured, a
#    deterministic offline stand-in otherwise. Isolated in its own function
#    so the graph node (and basic error handling) stays simple.
# ---------------------------------------------------------------------------
def _respond(messages: list[BaseMessage]) -> AIMessage:
    """Return an AI reply to `messages` (already trimmed + system-prefixed)."""
    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
            result = llm.invoke(messages)
            return AIMessage(content=result.content)
        except Exception as exc:  # network error, bad key, rate limit, etc.
            print(f"(Live Groq call failed: {exc!r} — falling back to offline stand-in.)")

    # Offline / no-API-key fallback: a free, local, deterministic "model"
    # so the whole graph + checkpointer pipeline is still runnable without
    # any network access or paid service.
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )
    return AIMessage(content=f"[offline stand-in] You said: {last_human!r}. (Set GROQ_API_KEY for a real reply.)")


# ---------------------------------------------------------------------------
# 2. The graph node: trims history, prepends the system prompt, responds.
# ---------------------------------------------------------------------------
def call_model(state: MessagesState) -> dict:
    """LangGraph node: read the full stored history from state, trim it to
    a budget, get a reply, and return it to be appended to state."""
    trimmed = trim_messages(
        state["messages"],
        max_tokens=6,  # token_counter below counts "1 message = 1 token"
        token_counter=len,
        strategy="last",
    )
    reply = _respond([SYSTEM_PROMPT, *trimmed])
    return {"messages": [reply]}


def build_app():
    """Compile a one-node graph with in-memory checkpointing, so repeated
    `.invoke()` calls sharing a thread_id resume that conversation."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    return workflow.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 3. Demo: memory persists within a thread_id, and is isolated across
#    different thread_ids.
# ---------------------------------------------------------------------------
def run_demo() -> None:
    app = build_app()

    thread_1 = {"configurable": {"thread_id": "demo-user-1"}}

    print("--- Turn 1 (thread 1) ---")
    out1 = app.invoke({"messages": [HumanMessage(content="My favorite programming language is Python. Remember that.")]}, config=thread_1)
    print("AI:", out1["messages"][-1].content)

    print("\n--- Turn 2 (thread 1, follow-up that relies on memory) ---")
    out2 = app.invoke({"messages": [HumanMessage(content="What is my favorite programming language?")]}, config=thread_1)
    print("AI:", out2["messages"][-1].content)
    print(f"(thread 1 now holds {len(out2['messages'])} total messages)")

    print("\n--- Turn 1 of a DIFFERENT thread (must have no memory of thread 1) ---")
    thread_2 = {"configurable": {"thread_id": "demo-user-2"}}
    out3 = app.invoke({"messages": [HumanMessage(content="What is my favorite programming language?")]}, config=thread_2)
    print("AI:", out3["messages"][-1].content)
    print(f"(thread 2 holds {len(out3['messages'])} total message(s), isolated from thread 1)")

    # Prove the checkpointer, not just an in-process list, is what's
    # carrying the state: fetch thread 1's persisted state directly.
    snapshot = app.get_state(thread_1)
    print(f"\nCheckpointer confirms thread 1's persisted state has {len(snapshot.values['messages'])} messages.")


# ---------------------------------------------------------------------------
# 4. Exercise: tune the trim budget and watch older turns drop out of what
#    the model sees, even though the checkpointer keeps the full history.
# ---------------------------------------------------------------------------
def trimming_exercise() -> None:
    print("\n--- Exercise: a longer conversation, trimmed to different budgets ---")

    long_history = [
        HumanMessage(content="Let's talk about LangChain."),
        AIMessage(content="Sure, what would you like to know?"),
        HumanMessage(content="What's a Runnable?"),
        AIMessage(content="A composable unit of a chain, like prompt | llm."),
        HumanMessage(content="And RunnableParallel?"),
        AIMessage(content="Runs several branches on the same input at once."),
    ]
    print(f"Full stored history: {len(long_history)} messages.")

    for budget in (2, 4, 10):
        trimmed = trim_messages(long_history, max_tokens=budget, token_counter=len, strategy="last")
        print(f"  max_tokens={budget:>2} -> {len(trimmed)} message(s) sent to the model: "
              f"{[m.content for m in trimmed]}")

    # Basic error handling: trim_messages rejects an invalid budget.
    try:
        trim_messages(long_history, max_tokens=-1, token_counter=len, strategy="last")
    except Exception as exc:
        print(f"\n(Expected) trim_messages rejects a negative budget: {exc!r}")


if __name__ == "__main__":
    run_demo()
    trimming_exercise()
