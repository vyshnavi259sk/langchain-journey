"""
Day 16: Long-Term Memory -- LangGraph's Store for cross-thread user memory
===========================================================================

Day 7 introduced LangGraph's checkpointer, and Day 15 made it durable with
`SqliteSaver`. Both of those solve the same problem: "remember the state of
*this one conversation* (`thread_id`) so it can resume later." But a
checkpoint is scoped to a single thread on purpose -- ask the same agent a
question on a brand-new `thread_id` and it has no idea the user ever
existed, even if yesterday's thread taught it the user's name and
preferences. Today's topic is LangGraph's other persistence primitive, the
**`Store`**, which is deliberately scoped the opposite way: across every
thread, keyed by something like a `user_id` instead of a `thread_id`.

What the concept is: a store is a flat, namespaced key-value database --
`store.put(namespace, key, value)`, `store.get(namespace, key)`,
`store.search(namespace)`, `store.delete(namespace, key)` -- where
`namespace` is a tuple such as `("memories", user_id)`. Unlike checkpointer
state (which belongs to one `thread_id` and holds a whole graph's messages),
a store holds small, deliberately-chosen facts an agent decides to save
("the user prefers metric units"), and nothing stops two completely
different `thread_id`s from reading and writing the *same* namespace, as
long as they agree on the `user_id`. That's what makes it "long-term":
memory that outlives any one conversation.

Important components and how they work:

- **`BaseStore`** (`langgraph.store.base`) -- the interface every store
  backend implements: `put`/`get`/`search`/`delete`, each keyed by
  `(namespace, key)`. `InMemoryStore` (`langgraph.store.memory`) keeps
  everything in a Python dict, same trade-off as Day 7's `InMemorySaver` --
  simple, but gone the moment the process exits.
- **`StateGraph.compile(store=...)`** -- the store is wired into a graph
  exactly like a checkpointer (Day 7/15: `compile(checkpointer=...)`). A
  node that declares a `store: BaseStore` parameter gets the store injected
  automatically by LangGraph -- no need to thread it through state or config
  by hand.
- **`SqliteStore.from_conn_string(path)`** (`langgraph.store.sqlite`) --
  a drop-in durable backend, the store's equivalent of Day 15's
  `SqliteSaver`. Same lesson as that day: swapping the backend is a one-line
  infrastructure change, and two independent connections to the same file
  see the same data, which is exactly what "long-term" needs to mean in
  practice (the agent process can restart and still recall old users).
- **Namespace-as-isolation** -- `("memories", "u-asha")` and
  `("memories", "u-rahul")` never see each other's data. This is the natural
  way a store keeps different users' long-term memories apart, the same way
  `thread_id` keeps conversations apart for a checkpointer.

Why it matters: a support bot, personal assistant, or any agent used across
multiple sessions needs *some* facts to survive past the conversation that
produced them -- a name, a stated preference, a past decision -- without
replaying the entire message history of every prior thread into the prompt
each time (which doesn't scale and leaks irrelevant context). The store is
LangGraph's answer to that: a small, explicitly-managed long-term memory
that sits *beside* the per-thread checkpointer rather than inside it.

The practical implementation, `day16_long_term_memory.py`, builds a tiny
two-node agent (`remember` -- saves any stated preference as a store fact;
`respond` -- answers using whatever facts the store has for that
`user_id`, regardless of which `thread_id` is asking) backed by `ChatGroq`
for extraction/answering when `GROQ_API_KEY` is set, and a deterministic
keyword-based extractor/responder otherwise (same offline-first pattern as
every agent day since Day 9). The demo: (1) thread "session-1" tells the
agent a preference; (2) a brand-new, unrelated `thread_id` ("session-2") for
the *same* `user_id` asks a question and the agent recalls the preference
anyway, proving memory crossed threads; (3) a *different* `user_id` asks the
same question on its own thread and gets no stored preference, proving
per-user isolation. It then repeats the whole flow against a `SqliteStore`
file, closing and reopening a fresh connection between steps (the Day-15
two-process trick, applied to the store instead of the checkpointer) to
prove the memory survives a process restart, not just the one Python
session. The exercise deletes a stored preference and confirms `store.get`
returns `None` afterwards (not an exception) and that `respond` falls back
to its no-memory answer -- plus a direct `store.get` on a key that was never
written, confirming the "missing key" case is handled the same way in both
backends. No new environment variables -- `GROQ_API_KEY` is reused exactly
as in Days 6-15. No new dependency: `langgraph.store.memory` and
`langgraph.store.sqlite` both ship with `langgraph`/`langgraph-checkpoint-sqlite`,
already in `requirements.txt` since Days 7/15.
"""

import os
import re
import tempfile
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    from langchain_groq import ChatGroq

    _llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
else:
    _llm = None


# ---------------------------------------------------------------------------
# Preference extraction -- "did this message state a preference worth
# remembering, and what is it?" Real LLM when available, a small regex-based
# stand-in otherwise -- the same offline-first contract as every agent day
# since Day 9.
# ---------------------------------------------------------------------------
_PREFERENCE_PATTERN = re.compile(
    r"\bi (?:prefer|like|use) (.+?)(?:\.|$)", re.IGNORECASE
)


def extract_preference(text: str) -> Optional[str]:
    """Returns a short preference string if `text` states one, else None."""
    if _llm is not None:
        prompt = (
            "If the following message states a personal preference the "
            "assistant should remember for later, reply with just that "
            "preference in under 8 words. If it states no preference, "
            f"reply with exactly NONE.\n\nMessage: {text}"
        )
        reply = _llm.invoke(prompt).content.strip()
        return None if reply.upper() == "NONE" else reply

    match = _PREFERENCE_PATTERN.search(text)
    return match.group(1).strip() if match else None


def answer_with_memory(text: str, remembered: Optional[str]) -> str:
    """Answers `text`, using `remembered` (a prior preference, or None)."""
    if _llm is not None:
        context = f"Known preference: {remembered}" if remembered else "No known preference."
        prompt = (
            f"{context}\nUser message: {text}\n"
            "Reply in one short sentence, using the known preference if "
            "relevant and if one is known."
        )
        return _llm.invoke(prompt).content.strip()

    if remembered:
        return f"Sure -- I remember you prefer {remembered}, so I'll go with that."
    return "I don't have any stored preference for you yet."


# ---------------------------------------------------------------------------
# Graph: two nodes, both store-aware. `user_id` (NOT `thread_id`) is the key
# the store is namespaced on, so the same `user_id` is recognized no matter
# which thread is currently talking to it.
# ---------------------------------------------------------------------------
def remember(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    last_human = state["messages"][-1].content
    preference = extract_preference(last_human)
    if preference:
        store.put(("memories", user_id), "preference", {"value": preference})
    return {}


def respond(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"]["user_id"]
    last_human = state["messages"][-1].content
    item = store.get(("memories", user_id), "preference")
    remembered = item.value["value"] if item is not None else None
    reply = answer_with_memory(last_human, remembered)
    return {"messages": [AIMessage(content=reply)]}


def build_graph() -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node("remember", remember)
    builder.add_node("respond", respond)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "respond")
    builder.add_edge("respond", END)
    return builder


def ask(graph, thread_id: str, user_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Part 1: InMemoryStore -- proves cross-thread recall and per-user isolation
# within a single process.
# ---------------------------------------------------------------------------
def demo_in_memory_store() -> None:
    print("=== Part 1: InMemoryStore -- cross-thread recall, per-user isolation ===")
    store = InMemoryStore()
    graph = build_graph().compile(checkpointer=InMemorySaver(), store=store)

    # Asha states a preference on "session-1".
    reply1 = ask(graph, thread_id="session-1", user_id="u-asha", text="I prefer metric units.")
    print(f"  [session-1, u-asha] 'I prefer metric units.' -> {reply1!r}")

    # A *different* thread_id, *same* user_id, asks a plain question with no
    # stated preference. The checkpointer has zero memory of session-1 (it's
    # a different thread_id entirely) -- any recall here can only be coming
    # from the store.
    reply2 = ask(graph, thread_id="session-2", user_id="u-asha", text="What units should I use?")
    print(f"  [session-2, u-asha] 'What units should I use?' -> {reply2!r}")
    assert "metric" in reply2.lower(), "expected session-2 to recall session-1's preference"
    print("  confirmed: session-2 recalled a preference stated on session-1 -- "
          "cross-thread memory via the store, not the checkpointer.\n")

    # A different user, own thread, same question: must NOT see Asha's data.
    reply3 = ask(graph, thread_id="session-3", user_id="u-rahul", text="What units should I use?")
    print(f"  [session-3, u-rahul] 'What units should I use?' -> {reply3!r}")
    assert "metric" not in reply3.lower(), "user isolation broken: u-rahul saw u-asha's preference"
    print("  confirmed: u-rahul's own thread sees no preference -- namespaces "
          "isolate different users on the same store.\n")


# ---------------------------------------------------------------------------
# Part 2: SqliteStore -- the Day-15 "close one connection, open a fresh one"
# trick, applied to the store, to prove memory survives a process restart.
# ---------------------------------------------------------------------------
def demo_sqlite_store_across_restarts() -> None:
    print("=== Part 2: SqliteStore -- memory survives a simulated process restart ===")
    tmp_dir = tempfile.mkdtemp(prefix="langgraph_day16_")
    db_path = os.path.join(tmp_dir, "memory.sqlite")
    print(f"  Using a throwaway SQLite file for this demo: {db_path}")

    try:
        # "Process A": save a preference, then close this connection entirely.
        with SqliteStore.from_conn_string(db_path) as store:
            store.setup()
            graph = build_graph().compile(checkpointer=InMemorySaver(), store=store)
            reply = ask(graph, thread_id="s1", user_id="u-asha", text="I like dark mode.")
            print(f"  [process A, s1, u-asha] 'I like dark mode.' -> {reply!r}")
        # The `with` block above closed the SQLite connection -- nothing
        # from that `store`/`graph` object is referenced again below.

        # "Process B": a brand-new SqliteStore connection against the same
        # file, a brand-new graph object, and a thread_id ("s2") that never
        # existed in process A.
        with SqliteStore.from_conn_string(db_path) as store:
            graph = build_graph().compile(checkpointer=InMemorySaver(), store=store)
            reply = ask(graph, thread_id="s2", user_id="u-asha", text="How should I display this?")
            print(f"  [process B, s2, u-asha] 'How should I display this?' -> {reply!r}")
            assert "dark mode" in reply.lower(), (
                "expected the SQLite-backed store to recall process A's preference"
            )
            print("  confirmed: a fresh connection on a new thread still recalled "
                  "the preference written before the 'restart'.\n")
    finally:
        for name in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, name))
        os.rmdir(tmp_dir)


# ---------------------------------------------------------------------------
# Exercise: deleting a memory, and reading a key that was never written --
# both must behave the same way (return None, not raise) on both backends.
# ---------------------------------------------------------------------------
def exercise_delete_and_missing_key() -> None:
    print("=== Exercise: deleting a memory, and reading a never-written key ===")
    store = InMemoryStore()
    graph = build_graph().compile(checkpointer=InMemorySaver(), store=store)

    ask(graph, thread_id="s1", user_id="u-lee", text="I prefer tea over coffee.")
    before = store.get(("memories", "u-lee"), "preference")
    print(f"  before delete: {before.value if before else None}")
    assert before is not None

    store.delete(("memories", "u-lee"), "preference")
    after = store.get(("memories", "u-lee"), "preference")
    print(f"  after delete:  {after}")
    assert after is None, "expected store.get to return None for a deleted key"

    reply = ask(graph, thread_id="s2", user_id="u-lee", text="What should I drink?")
    print(f"  [s2, u-lee, post-delete] 'What should I drink?' -> {reply!r}")
    assert "tea" not in reply.lower(), "deleted preference should no longer be recalled"

    # A key that was *never* written for a brand-new user -- same contract:
    # None, no exception, no KeyError.
    never_written = store.get(("memories", "u-new-user"), "preference")
    print(f"  store.get on a never-written key: {never_written!r}")
    assert never_written is None

    print("  confirmed: a deleted key and a never-written key both return "
          "None -- no special-casing needed by callers either way.\n")


def main() -> None:
    demo_in_memory_store()
    demo_sqlite_store_across_restarts()
    exercise_delete_and_missing_key()
    print("All scenarios completed. Cross-thread recall, per-user isolation, "
          "restart durability, and delete/missing-key handling all verified "
          "end-to-end" + (" (offline, no GROQ_API_KEY set)." if _llm is None else "."))


if __name__ == "__main__":
    main()
