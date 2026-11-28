"""
Day 17: Streaming -- LangGraph's `.stream()` / `.astream()` and stream modes
=============================================================================

Every agent built since Day 9 (tool calling) has been run with `.invoke()`:
send a full input, block, get a full output back. That is fine for a script,
but it is a bad user experience for anything with a human watching a screen --
a multi-second agent turn with retrieval + an LLM call feels frozen until the
very last token arrives. Today's topic is LangGraph's **streaming** interface,
which turns that single blocking call into an iterator of incremental events
as the graph actually executes.

What the concept is
--------------------
`graph.stream(input, config, stream_mode=...)` (and its async twin
`graph.astream(...)`) return an iterator/async-iterator of events instead of
a single final state. `stream_mode` controls *what* each event contains:

- `"values"`   -- the full graph state after each super-step (node run).
- `"updates"`  -- only the *delta* each node returned, keyed by node name.
- `"messages"` -- `(message_chunk, metadata)` tuples streamed token-by-token
  from any chat model invoked inside a node -- this fires even if the node
  calls the model with a plain `.invoke()`, because LangGraph instruments the
  model's own streaming callbacks automatically.
- `"custom"`   -- arbitrary data a node pushes itself via
  `langgraph.config.get_stream_writer()`, independent of the graph's state --
  the mechanism for "searching... found 2 of 5 documents"-style progress
  that has nothing to do with the final answer.

`stream_mode` can also be a *list* of modes (e.g. `["updates", "messages"]`),
in which case each event becomes a `(mode, payload)` tuple -- this is the
realistic shape for a chat UI: `"messages"` renders the answer live while
`"updates"` drives a separate "thinking..." progress indicator.

Important components and how they work
----------------------------------------
- `get_stream_writer()` -- called with no arguments *inside* a running node,
  returns a callable; calling that callable with any JSON-able value emits
  one `"custom"` stream event immediately, without touching graph state.
- `FakeListChatModel` (`langchain_core.language_models.fake_chat_models`) --
  a real LangChain testing utility that streams its canned response one
  character at a time via `.stream()`/`.invoke()`, exactly like a real chat
  model would stream tokens. It is today's free/local stand-in for
  `ChatGroq` when `GROQ_API_KEY` isn't set, so `stream_mode="messages"` has
  something genuine to stream offline -- not a hand-rolled string splitter.
- Everything else (`StateGraph`, `MessagesState`, `InMemorySaver`) is exactly
  Day 7/15/16's checkpointing machinery, reused unchanged -- streaming is an
  orthogonal concern layered on top of the same compiled graph.

Why it matters
---------------
Streaming is what makes a multi-second agent turn *feel* instant: the first
token (or the first "searching..." progress event) can appear in well under
a second even if the full answer takes five. It is also the only practical
way to drive a real-time chat UI or a websocket/SSE API endpoint -- you
cannot hold an HTTP response open and send nothing for five seconds and call
it a good product. Every agent built on Days 9-16 could be wrapped in this
same `.stream()` call with no change to the graph itself.

The practical implementation, `day17_streaming.py`, builds a small three-node
graph (`plan -> research -> answer`): `plan_node` extracts a search query
from the user's question (deterministic, no LLM needed); `research_node`
scores a tiny local corpus of one-line blurbs about Days 1-16 of this project
by keyword overlap -- a legitimate free/local "retrieval" stand-in -- and
emits a `"custom"` progress event per candidate it scores; `answer_node`
calls `ChatGroq` when `GROQ_API_KEY` is set, or `FakeListChatModel` otherwise,
and appends the reply as an `AIMessage`. The demo runs the *same* compiled
graph under `"updates"`, `"values"`, `"custom"`, `"messages"`, and a
multiplexed `["updates", "messages"]` stream, and reconstructs the streamed
answer from the token chunks to assert it matches the final state's message
verbatim -- proving the tokens really are the answer, not a side channel.
The exercise adds an async `.astream()` run (how a websocket/SSE endpoint
would actually consume this) and a time-to-first-event vs. time-to-completion
comparison that makes the latency win concrete instead of just asserted.
Deliberate error handling: `plan_node` raises a clear `ValueError` on an
empty question instead of crashing on an index error, and `research_node`
falls back to an explicit "no sources found" path (still a valid, if
unsupported, answer) rather than crashing when nothing in the corpus
matches. No new environment variables -- `GROQ_API_KEY` is reused exactly as
in Days 6-16. New dependency: none beyond what Days 7/9 already added
(`langgraph`, `langchain-core`); `FakeListChatModel` ships with
`langchain-core`, already in `requirements.txt`.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


# ---------------------------------------------------------------------------
# A tiny local "corpus" -- one-line blurbs about this project's own earlier
# days, standing in for a real document store (Day 5's FAISS index already
# covers that; today's focus is streaming, so keyword search keeps the demo
# self-contained and dependency-free).
# ---------------------------------------------------------------------------
CORPUS: list[tuple[str, str]] = [
    ("day7", "conversation memory uses a langgraph stategraph with a checkpointer keyed by thread id"),
    ("day9", "tool calling binds python functions to a chat model so it can call them on demand"),
    ("day10", "the react agent pattern loops reason act observe using create react agent"),
    ("day11", "agentic rag lets the agent decide per question whether retrieval is even needed"),
    ("day12", "a multi agent supervisor classifies each question and routes to a specialist agent"),
    ("day13", "agent handoffs use the command object to update state and route in one return value"),
    ("day14", "human in the loop uses interrupt to pause a graph until a person approves a tool call"),
    ("day15", "sqlite saver makes checkpoints durable on disk so a paused thread survives a restart"),
    ("day16", "the long term memory store keys facts by user id instead of thread id across sessions"),
]


# A small stopword list so "keyword overlap" measures topical relevance
# instead of counting shared articles and prepositions -- without this,
# almost any English question "matches" almost any blurb on words like
# "a", "the", or "of".
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "of", "to",
    "and", "or", "how", "what", "did", "do", "does", "handle", "for",
    "with", "that", "this", "it", "about", "tell", "me", "not", "no",
}


class ResearchState(MessagesState):
    """MessagesState plus two fields this graph's nodes pass along."""

    query: str
    sources: list[str]


def plan_node(state: ResearchState) -> dict:
    """Extract a lowercase search query from the latest human message.

    Deliberate error handling: a graph invoked with no messages at all is a
    caller bug, not something to paper over with an empty string -- raise a
    clear ValueError instead of letting `messages[-1]` crash with a confusing
    IndexError three frames down.
    """
    messages = state["messages"]
    if not messages:
        raise ValueError("plan_node requires at least one message to plan a query from")
    query = str(messages[-1].content).strip().lower()
    if not query:
        raise ValueError("plan_node requires a non-empty question")
    return {"query": query}


def research_node(state: ResearchState) -> dict:
    """Score the local CORPUS by keyword overlap, streaming progress as we go.

    `get_stream_writer()` is only meaningful while a graph is actually
    streaming; called during a plain `.invoke()` it safely returns a no-op
    writer, so this node works identically either way.
    """
    writer = get_stream_writer()
    query_words = {w for w in state["query"].split() if w not in STOPWORDS}

    scored: list[tuple[int, str, str]] = []
    for day, blurb in CORPUS:
        overlap = len(query_words & set(blurb.split()))
        writer({"event": "scored", "day": day, "overlap": overlap})
        if overlap > 0:
            scored.append((overlap, day, blurb))

    if not scored:
        writer({"event": "no_match", "message": "no corpus entry overlapped the query"})
        return {"sources": []}

    scored.sort(reverse=True)
    top = scored[:2]
    writer({"event": "done", "picked": [day for _, day, _ in top]})
    return {"sources": [f"{day}: {blurb}" for _, day, blurb in top]}


def _build_llm():
    """Real ChatGroq when a key is configured, a deterministic offline
    streaming stand-in otherwise. Both expose the same `.invoke()`/`.stream()`
    interface, so `answer_node` below never needs to know which one it has.
    """
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return FakeListChatModel(
        responses=[
            "Based on this project's own notes, that topic connects to prior "
            "LangGraph building blocks already covered -- see the sources below."
        ]
    )


def answer_node(state: ResearchState) -> dict:
    """Compose a short grounded answer from whatever `research_node` found."""
    llm = _build_llm()
    if state["sources"]:
        context = "\n".join(f"- {s}" for s in state["sources"])
        system = SystemMessage(
            content=f"Answer briefly using only these notes:\n{context}"
        )
    else:
        system = SystemMessage(
            content="No matching notes were found; say so plainly and answer briefly anyway."
        )
    reply = llm.invoke([system, *state["messages"]])
    # Return the model's own response object, not a freshly-built AIMessage
    # with the same text: LangGraph's "messages" stream mode ties streamed
    # token chunks to the response object's own id/run metadata, so wrapping
    # the content in a brand-new AIMessage here would make the graph treat
    # it as a second, already-complete message -- doubling every answer.
    return {"messages": [reply]}


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("answer", answer_node)
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "research")
    workflow.add_edge("research", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Demos: the same compiled graph, run under each stream_mode in turn.
# ---------------------------------------------------------------------------
def demo_updates_mode(graph, question: str, thread_id: str) -> None:
    print(f"\n--- stream_mode='updates' ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    for update in graph.stream(inputs, config, stream_mode="updates"):
        for node_name, delta in update.items():
            print(f"  [{node_name}] -> {delta}")


def demo_values_mode(graph, question: str, thread_id: str) -> None:
    print(f"\n--- stream_mode='values' ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    for i, snapshot in enumerate(graph.stream(inputs, config, stream_mode="values")):
        keys = {k: v for k, v in snapshot.items() if k != "messages"}
        print(f"  step {i}: query/sources so far -> {keys}")


def demo_custom_mode(graph, question: str, thread_id: str) -> None:
    print(f"\n--- stream_mode='custom' ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    for event in graph.stream(inputs, config, stream_mode="custom"):
        print(f"  progress: {event}")


def demo_messages_mode(graph, question: str, thread_id: str) -> str:
    print(f"\n--- stream_mode='messages' ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    print("  tokens: ", end="", flush=True)
    collected = []
    for chunk, metadata in graph.stream(inputs, config, stream_mode="messages"):
        if chunk.content and metadata.get("langgraph_node") == "answer":
            print(chunk.content, end="", flush=True)
            collected.append(chunk.content)
    print()
    return "".join(collected)


def demo_multiplexed_stream(graph, question: str, thread_id: str) -> None:
    """The realistic shape for a chat UI: progress events and answer tokens
    arriving on the same iterator, told apart by the `mode` tag."""
    print(f"\n--- stream_mode=['updates', 'messages'] ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    for mode, payload in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
        if mode == "updates":
            for node_name in payload:
                print(f"  [progress] {node_name} finished")
        elif mode == "messages":
            chunk, metadata = payload
            if chunk.content and metadata.get("langgraph_node") == "answer":
                print(chunk.content, end="", flush=True)
    print()


async def demo_async_stream(graph, question: str, thread_id: str) -> str:
    """How a websocket/SSE endpoint would actually consume this: async
    iteration over `.astream()` instead of blocking the event loop."""
    print(f"\n--- astream(), stream_mode='messages' ({question!r}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}
    collected = []
    async for chunk, metadata in graph.astream(inputs, config, stream_mode="messages"):
        if chunk.content and metadata.get("langgraph_node") == "answer":
            collected.append(chunk.content)
    text = "".join(collected)
    print(f"  (received {len(collected)} chunks asynchronously): {text}")
    return text


def demo_latency_comparison(graph, question: str) -> None:
    """Make the streaming win concrete: time to *first* event vs. time to
    the full answer, for the same question run two different ways."""
    print(f"\n--- latency: invoke() vs. stream() first-event ({question!r}) ---")

    config = {"configurable": {"thread_id": "latency-invoke"}}
    inputs = {"messages": [HumanMessage(content=question)]}
    start = time.perf_counter()
    graph.invoke(inputs, config)
    invoke_total = time.perf_counter() - start

    config = {"configurable": {"thread_id": "latency-stream"}}
    start = time.perf_counter()
    first_event_at = None
    for _ in graph.stream(inputs, config, stream_mode="custom"):
        if first_event_at is None:
            first_event_at = time.perf_counter() - start
    stream_total = time.perf_counter() - start

    print(f"  invoke(): no output until {invoke_total:.4f}s (the whole turn)")
    print(f"  stream(): first progress event at {first_event_at:.4f}s, full turn {stream_total:.4f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY found -- answer_node will call the real ChatGroq model.\n")
    else:
        print(
            "GROQ_API_KEY not set -- answer_node will use FakeListChatModel, a real "
            "LangChain testing utility that streams a canned response character by "
            "character, so every stream_mode below still has something genuine to "
            "stream.\n"
        )

    graph = build_graph()
    question = "How did day 13 handle agent handoffs?"

    demo_updates_mode(graph, question, thread_id="demo-updates")
    demo_values_mode(graph, question, thread_id="demo-values")
    demo_custom_mode(graph, question, thread_id="demo-custom")
    streamed_answer = demo_messages_mode(graph, question, thread_id="demo-messages")
    demo_multiplexed_stream(graph, question, thread_id="demo-multiplexed")

    # Sanity check: the tokens streamed under "messages" mode must reconstruct
    # to exactly the final AIMessage stored in that thread's checkpointed state.
    final_state = graph.get_state({"configurable": {"thread_id": "demo-messages"}})
    final_answer = final_state.values["messages"][-1].content
    assert streamed_answer == final_answer, "streamed tokens must match the final message"
    print("\n[assert] streamed tokens reconstruct the final AIMessage exactly -- OK")

    # Exercise, part 1: the async path a real API endpoint would use.
    async_answer = asyncio.run(demo_async_stream(graph, question, thread_id="demo-async"))
    assert async_answer, "astream() should have yielded at least one token chunk"
    print("[assert] astream() yielded a non-empty answer -- OK")

    # Exercise, part 2: make the latency benefit concrete, not just asserted.
    demo_latency_comparison(graph, "Tell me about day 16.")

    # Deliberate error-handling checks, offline, no network call either way.
    print("\n--- error handling: empty question ---")
    try:
        graph.invoke({"messages": []}, {"configurable": {"thread_id": "demo-error"}})
    except ValueError as exc:
        print(f"  caught expected ValueError: {exc}")

    print("\n--- error handling: question with no matching corpus entry ---")
    no_match_result = graph.invoke(
        {"messages": [HumanMessage(content="what is the capital of a country not in this project")]},
        {"configurable": {"thread_id": "demo-no-match"}},
    )
    assert no_match_result["sources"] == [], "an off-topic question should find zero corpus matches"
    print(f"  sources found: {no_match_result['sources']!r} (empty, as expected)")
    print(f"  answer still produced: {no_match_result['messages'][-1].content[:80]}...")


if __name__ == "__main__":
    main()
