"""
Day 15: Persistent Checkpointing -- LangGraph's SqliteSaver for durable state
==============================================================================

Day 7 introduced LangGraph's checkpointer concept via `InMemorySaver`
(`MemorySaver`), and Day 14 leaned on it hard: `interrupt()` only works
because the checkpointer durably records exactly where a graph paused, so a
human can approve or reject a pending action later. But "later" in Day 14
still meant "later in the same Python process" -- `InMemorySaver` keeps every
checkpoint in a plain Python dict, so the moment the process exits, every
paused thread is gone. A real approval workflow needs the service to be
allowed to *restart* between "agent asks for approval" and "human approves
it" -- a deploy, a crash, a scheduled restart -- without losing the pending
request. Today's topic is LangGraph's **`SqliteSaver`**, a drop-in
checkpointer that writes the same checkpoints to a SQLite file on disk
instead of to RAM.

What the concept is: a checkpointer is anything implementing LangGraph's
`BaseCheckpointSaver` interface -- `put`/`get`/`list`, keyed by
`(thread_id, checkpoint_id)`. `StateGraph.compile(checkpointer=...)` is the
only line that changes between an in-memory graph and a persistent one; the
graph logic, `interrupt()`, and `Command(resume=...)` contract from Day 14
are completely unchanged. Swapping the checkpointer is an infrastructure
decision, not an application one.

Important components and how they work:

- **`SqliteSaver.from_conn_string(path)`** (`langgraph.checkpoint.sqlite`) --
  a context manager that opens (and creates, if missing) a SQLite database
  at `path` and yields a checkpointer backed by it. Every `put` LangGraph
  issues (one per super-step) is an `INSERT`/`UPDATE` against that file, so
  the data survives the Python process exiting.
- **`thread_id`** -- the key every checkpoint is filed under. As long as two
  processes point `SqliteSaver.from_conn_string` at the *same file path* and
  use the *same `thread_id`*, the second process can resume exactly where
  the first one paused, even though it never shared memory with it.
- **`graph.get_state_history(config)`** -- walks every checkpoint ever
  recorded for a thread, oldest last, newest first. This is what makes a
  SQLite-backed (or Postgres-backed) checkpointer more than just "memory
  that happens to survive a restart": the full history of a thread's state
  transitions is queryable, which is the basis for LangGraph's "time
  travel" (replaying or forking from an older checkpoint).
- **`Command(resume=...)`** (Day 14, unchanged) -- still how a paused thread
  is continued; it now reads its pending `interrupt()` payload out of the
  SQLite file instead of a dict that lived in the first process's heap.

Why it matters: this is the gap between "human-in-the-loop looks correct in
a demo" and "human-in-the-loop survives a Tuesday-afternoon deploy." Any
agent that pauses for approval, runs long enough to outlive a worker
process, or needs an audit trail of exactly how its state evolved needs a
checkpointer backed by real storage -- SQLite here as the simplest durable
option, with the identical API available for Postgres
(`langgraph.checkpoint.postgres.PostgresSaver`) in production. Because the
checkpointer is swapped via one constructor call, nothing else written since
Day 7 has to change to make that jump.

The practical implementation, `day15_persistent_checkpointing.py`, builds a
tiny three-node "expense approval" graph (`submit -> request_approval ->
apply`) that calls `interrupt()` in `request_approval`, structurally
identical to Day 14's pattern. It proves persistence by (1) opening a
`SqliteSaver` against a file, invoking the graph until it pauses on
`interrupt()`, then *closing that connection entirely* -- simulating the
first process exiting; (2) opening a *brand-new* `SqliteSaver` connection
against the same file (a fresh object, with no Python references back to
the first run) and resuming the same `thread_id` with `Command(resume=...)`,
which only succeeds because the pending interrupt was read back off disk;
and (3) calling `get_state_history()` on the resumed thread to print every
checkpoint recorded, oldest first. The exercise re-runs the same two-process
pattern for a *rejected* request and a *second, independent* `thread_id` to
confirm threads never cross-contaminate on a shared SQLite file. No new
environment variables. New dependency: `langgraph-checkpoint-sqlite`, added
to `requirements.txt` (the `langgraph.checkpoint.sqlite` import is not part
of the core `langgraph` package).
"""

import os
import tempfile
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# Graph definition -- a minimal expense-approval workflow.
#
# submit            -- records the request.
# request_approval  -- pauses the whole graph via interrupt() until a human
#                      (or, in this demo, a second simulated process) calls
#                      graph.invoke(Command(resume=...)) on the same thread.
# apply             -- acts on the human's decision.
# ---------------------------------------------------------------------------
class ExpenseState(TypedDict):
    amount: float
    memo: str
    status: str
    log: list[str]


def submit(state: ExpenseState) -> dict:
    return {"log": state["log"] + [f"submitted ${state['amount']:.2f} ({state['memo']})"]}


def request_approval(state: ExpenseState) -> dict:
    # Pauses here. On first entry this blocks the graph and surfaces the
    # payload to the caller. On a *resumed* entry (even from a different
    # process, different SqliteSaver object, possibly minutes or days later)
    # interrupt() returns the value passed to Command(resume=...) instead of
    # pausing again.
    decision = interrupt(
        {
            "question": "Approve this expense?",
            "amount": state["amount"],
            "memo": state["memo"],
        }
    )
    approved = bool(decision)
    return {
        "status": "approved" if approved else "rejected",
        "log": state["log"] + [f"human {'approved' if approved else 'rejected'} the request"],
    }


def apply(state: ExpenseState) -> dict:
    if state["status"] == "approved":
        action = f"reimbursed ${state['amount']:.2f} for '{state['memo']}'"
    else:
        action = f"no payment issued for '{state['memo']}' (rejected)"
    return {"log": state["log"] + [action]}


def build_graph() -> StateGraph:
    builder = StateGraph(ExpenseState)
    builder.add_node("submit", submit)
    builder.add_node("request_approval", request_approval)
    builder.add_node("apply", apply)
    builder.add_edge(START, "submit")
    builder.add_edge("submit", "request_approval")
    builder.add_edge("request_approval", "apply")
    builder.add_edge("apply", END)
    return builder


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------
def run_until_pause(db_path: str, thread_id: str, amount: float, memo: str) -> dict:
    """Simulates "process A": opens a fresh SQLite connection, invokes the
    graph until it hits interrupt(), then closes the connection and returns.
    Nothing from this function's local state is reused afterwards -- only
    the file at db_path and the thread_id carry over.
    """
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(
            {"amount": amount, "memo": memo, "status": "pending", "log": []},
            config=config,
        )
        pending = result.get("__interrupt__")
        if not pending:
            raise RuntimeError(f"expected an interrupt but graph finished: {result}")
        print(f"  [process A] paused -- pending approval: {pending[0].value}")
    # The `with` block exit here closes the SQLite connection. Nothing from
    # `checkpointer` or `graph` above is referenced again below.
    return result


def resume_from_disk(db_path: str, thread_id: str, approve: bool) -> dict:
    """Simulates "process B": a brand-new SqliteSaver connection (and a
    brand-new graph object -- same code, different Python object) against
    the same file. It has no in-memory link whatsoever to process A; the
    only shared state is the SQLite file and the thread_id.
    """
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(Command(resume=approve), config=config)

        print("  [process B] resumed from disk, final log:")
        for line in final_state["log"]:
            print(f"    - {line}")

        print("  [process B] full checkpoint history for this thread (oldest first):")
        history = list(graph.get_state_history(config))
        for snapshot in reversed(history):
            step = snapshot.metadata.get("step")
            print(f"    step {step}: status={snapshot.values.get('status')!r}")

    return final_state


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="langgraph_day15_")
    db_path = os.path.join(tmp_dir, "checkpoints.sqlite")
    print(f"Using a throwaway SQLite file for this demo: {db_path}\n")

    try:
        # --- Scenario 1: approved, across two independent connections -----
        print("Scenario 1: expense gets approved after a simulated restart")
        run_until_pause(db_path, thread_id="expense-1", amount=42.50, memo="team lunch")
        final1 = resume_from_disk(db_path, thread_id="expense-1", approve=True)
        assert final1["status"] == "approved"
        print()

        # --- Scenario 2: rejected, independent thread ----------------------
        print("Scenario 2: a *different* expense, on a *different* thread_id, "
              "gets rejected")
        run_until_pause(db_path, thread_id="expense-2", amount=999.00, memo="gold-plated keyboard")
        final2 = resume_from_disk(db_path, thread_id="expense-2", approve=False)
        assert final2["status"] == "rejected"
        print()

        # --- Exercise: confirm thread isolation on the shared file ---------
        # Re-resume expense-1 with a *different* checkpointer object again.
        # It must still report "approved" (from scenario 1's own resume),
        # proving expense-2's rejection never touched expense-1's record --
        # both live in the same SQLite file but under different thread_ids.
        print("Exercise: re-reading expense-1's final state from a third, "
              "fresh connection")
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = build_graph().compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "expense-1"}}
            snapshot = graph.get_state(config)
            status = snapshot.values["status"]
            print(f"  expense-1 status, read from a new connection: {status!r}")
            assert status == "approved", "thread isolation broken: expense-2's rejection leaked in"
        print("  confirmed: expense-1 and expense-2 never cross-contaminated.\n")

        print("All scenarios completed. Persistence, resume-after-restart, "
              "and thread isolation all verified end-to-end.")

    except Exception as exc:  # pragma: no cover - demo-time error reporting
        print(f"Demo failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        # This is a scratch file for the demo, not state we want to keep
        # around or accidentally commit -- clean it up either way.
        for name in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, name))
        os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
