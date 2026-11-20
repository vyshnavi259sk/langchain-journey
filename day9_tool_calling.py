"""
Day 9 -- Tool Calling: letting the LLM decide to call real Python functions
=============================================================================

Days 6/8 built RAG chains that ground answers in *static*, pre-indexed
documents -- great for "what does this project's config say", useless for
"what is 23 + 19" or anything that needs an exact computation or a live
side effect. Today's topic, **tool calling**, is the piece that lets an
LLM request that a specific Python function be run with specific
arguments, read back its real result, and use that to answer -- instead
of guessing or hallucinating a number.

This is also the direct foundation of the next few days on the roadmap
(agents, agent workflows): an "agent" is essentially this same
call -> execute -> feed-back loop, just repeated automatically by a
framework (e.g. LangGraph's `create_react_agent`) instead of hand-rolled
as it is here. Writing the loop by hand once, today, makes it much clearer
what an agent framework is actually automating later.

Key pieces introduced:

- **`langchain_core.tools.tool`** -- a decorator that turns an ordinary
  Python function into a `StructuredTool`. It reads the function's type
  hints to build a JSON-schema description of its arguments, and its
  docstring becomes the natural-language description the LLM sees when
  deciding whether (and how) to call it. A `BaseTool` exposes `.invoke(args)`
  to actually run it.
- **`ChatGroq.bind_tools(tools)`** -- returns a new Runnable wrapping the
  chat model with the tools' schemas attached. The model itself is
  unchanged; `bind_tools` just tells the provider's API "here are the
  functions you're allowed to request".
- **`AIMessage.tool_calls`** -- when a bound model decides a tool is
  needed, its response is an `AIMessage` with `content` often empty and
  `tool_calls` populated: a list of `{"name", "args", "id"}` dicts, one
  per requested call (a single turn can request more than one).
- **`ToolMessage`** -- after running a requested tool, its result is
  wrapped in a `ToolMessage(content=..., tool_call_id=...)` and appended
  to the message history. The `tool_call_id` is what lets the model match
  a result back to the specific call it made, which matters once more
  than one tool is called in the same turn.
- **The manual loop** -- `invoke -> if tool_calls: run them, append
  ToolMessages, invoke again -> else: done` -- repeated until the model
  stops requesting tools. This is exactly what `create_react_agent` (a
  likely topic soon) does internally.

Why it matters: retrieval (Day 6/8) answers from what was indexed ahead of
time; tool calling answers from what the model can *compute or fetch right
now*, and lets it decide for itself, per question, whether a tool is even
needed. Real products combine both -- a "search my docs" tool is itself
just another tool in this same loop (sketched in `search_project_notes`
below, a deliberately tiny stand-in for wrapping Day 6's real retriever as
a callable tool).

Like Day 7/8, this still runs fully offline when `GROQ_API_KEY` is not
set: a small deterministic keyword router stands in for the model's
tool-selection decision, so the whole call -> execute -> feed-back loop is
still demonstrably runnable end-to-end with no network call and no cost.
No new dependencies -- `langchain_core.tools` ships with `langchain-core`,
already in requirements.txt.
"""

import os
import sys
from typing import Any

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# 1. Tools. Each one is a plain, typed, documented Python function -- the
#    @tool decorator is doing all the schema-generation work for us.
# ---------------------------------------------------------------------------
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the exact sum."""
    return a + b


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a single word or short phrase."""
    return len(word)


@tool
def safe_divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator. Raises a clear error on division by zero
    instead of letting the caller crash, so the error can be reported back to
    the model as a tool result rather than an unhandled exception."""
    if denominator == 0:
        raise ValueError("Cannot divide by zero -- denominator must be non-zero.")
    return numerator / denominator


# A deliberately tiny stand-in for "retrieval as a tool": Day 6/8 built a
# real FAISS + TF-IDF retriever over a knowledge base; here the same *idea*
# -- letting the model pull in project-specific facts on demand -- is
# represented with a plain dict lookup so this file stays self-contained.
# Swapping this body for `retriever.invoke(query)` from day6_retrieval_rag.py
# would turn this into real "agentic RAG" with no other changes needed.
_PROJECT_NOTES = {
    "retriever": "This project's retriever (Day 5/6) is FAISS over TF-IDF vectors, k=2 by default.",
    "memory": "Conversation memory (Day 7) uses a LangGraph StateGraph with an InMemorySaver checkpointer.",
    "rag": "Conversational RAG (Day 8) rewrites follow-up questions before retrieving, so pronouns resolve correctly.",
}


@tool
def search_project_notes(topic: str) -> str:
    """Look up a short note about this project for a given topic keyword,
    e.g. 'retriever', 'memory', or 'rag'. Returns a message saying nothing
    was found if the topic keyword isn't recognized."""
    topic_key = topic.strip().lower()
    for key, note in _PROJECT_NOTES.items():
        if key in topic_key:
            return note
    return f"No project note found for topic '{topic}'."


TOOLS = [add_numbers, get_word_length, safe_divide, search_project_notes]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools for exact arithmetic, "
    "counting characters, and looking up short project notes. Use a tool "
    "whenever the question needs exact computation or project-specific facts "
    "instead of guessing. Once you have the tool results you need, give a "
    "concise final answer in plain language."
)


# ---------------------------------------------------------------------------
# 2. The model, bound to the tools -- or None if no API key is configured.
# ---------------------------------------------------------------------------
def _get_llm_with_tools():
    """Lazily builds a ChatGroq instance bound to TOOLS, or None without a key."""
    if not GROQ_API_KEY:
        return None
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)
        return llm.bind_tools(TOOLS)
    except Exception as exc:  # bad key, network error, package missing, etc.
        print(f"(Could not construct ChatGroq: {exc!r} -- falling back to offline router.)")
        return None


# ---------------------------------------------------------------------------
# 3. Offline fallback: a crude keyword router that mimics the model's job
#    of deciding which tool(s) to call and with what arguments, so the loop
#    below is exercised end-to-end with no network call. This is NOT a real
#    tool-calling model -- it only recognizes a couple of hand-written
#    question shapes well enough to demonstrate the mechanics.
# ---------------------------------------------------------------------------
def _offline_tool_router(question: str) -> AIMessage:
    import re

    question_lower = question.lower()
    calls: list[dict[str, Any]] = []

    add_match = re.search(r"(-?\d+\.?\d*)\s*(?:\+|plus)\s*(-?\d+\.?\d*)", question_lower)
    if add_match:
        calls.append(
            {
                "name": "add_numbers",
                "args": {"a": float(add_match.group(1)), "b": float(add_match.group(2))},
                "id": f"call_{len(calls)}",
            }
        )

    div_match = re.search(r"(-?\d+\.?\d*)\s*(?:/|divided by)\s*(-?\d+\.?\d*)", question_lower)
    if div_match:
        calls.append(
            {
                "name": "safe_divide",
                "args": {"numerator": float(div_match.group(1)), "denominator": float(div_match.group(2))},
                "id": f"call_{len(calls)}",
            }
        )

    letters_match = re.search(r"letters? (?:are )?in (?:the word )?['\"]?(\w+)['\"]?", question_lower)
    if letters_match:
        calls.append(
            {
                "name": "get_word_length",
                "args": {"word": letters_match.group(1)},
                "id": f"call_{len(calls)}",
            }
        )

    for topic_key in _PROJECT_NOTES:
        if topic_key in question_lower:
            calls.append({"name": "search_project_notes", "args": {"topic": topic_key}, "id": f"call_{len(calls)}"})
            break

    return AIMessage(content="", tool_calls=calls)


def _offline_final_answer(tool_messages: list[ToolMessage]) -> str:
    parts = [f"{m.name}(...) -> {m.content}" for m in tool_messages]
    return "[offline stand-in -- set GROQ_API_KEY for a real synthesized answer]\n  " + "\n  ".join(parts)


# ---------------------------------------------------------------------------
# 4. The manual tool-calling loop, shared by the demo and the exercise.
# ---------------------------------------------------------------------------
def run_tool_calling_loop(question: str, max_rounds: int = 3) -> str:
    """Runs the call -> execute -> feed-back loop until the model stops
    requesting tools (or max_rounds is hit, as a safety bound against an
    infinite loop), and returns the final text answer."""
    llm_with_tools = _get_llm_with_tools()
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    all_tool_messages: list[ToolMessage] = []

    for round_number in range(max_rounds):
        if llm_with_tools is not None:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as exc:  # network/auth failure mid-loop -- fall back
                print(f"(Live tool-calling call failed: {exc!r} -- using offline router.)")
                response = _offline_tool_router(question) if round_number == 0 else AIMessage(content="")
        else:
            # Only the offline router's *first* round proposes tool calls;
            # later rounds have nothing new to add, so it's done.
            response = _offline_tool_router(question) if round_number == 0 else AIMessage(content="")

        messages.append(response)

        if not response.tool_calls:
            if response.content:
                return response.content
            # Offline path never writes final prose itself -- synthesize one
            # from whatever tools actually ran this loop.
            return _offline_final_answer(all_tool_messages) if all_tool_messages else "(no answer produced)"

        for call in response.tool_calls:
            selected_tool = TOOLS_BY_NAME.get(call["name"])
            if selected_tool is None:
                result_content = f"Error: unknown tool '{call['name']}'."
            else:
                try:
                    result_content = str(selected_tool.invoke(call["args"]))
                except Exception as exc:  # basic error handling: report it, don't crash
                    result_content = f"Error running {call['name']}: {exc}"

            tool_message = ToolMessage(content=result_content, tool_call_id=call["id"], name=call["name"])
            all_tool_messages.append(tool_message)
            messages.append(tool_message)

        if llm_with_tools is None:
            # Offline router already produced every call it's going to in
            # round 0; synthesize the final answer now rather than looping
            # again with nothing new to say.
            return _offline_final_answer(all_tool_messages)

    return "(gave up after max_rounds without a final answer)"


# ---------------------------------------------------------------------------
# 5. Demo: one question needing a single tool, one needing two.
# ---------------------------------------------------------------------------
def run_demo() -> None:
    print("--- Single tool call ---")
    q1 = "What is 23 plus 19?"
    print(f"Q: {q1}\nA: {run_tool_calling_loop(q1)}")

    print("\n--- Two tool calls in one question ---")
    q2 = "What is 23 plus 19, and how many letters are in the word 'retrieval'?"
    print(f"Q: {q2}\nA: {run_tool_calling_loop(q2)}")

    print("\n--- A question needing a project-note lookup tool ---")
    q3 = "Tell me about this project's memory setup."
    print(f"Q: {q3}\nA: {run_tool_calling_loop(q3)}")


# ---------------------------------------------------------------------------
# 6. Exercise: force a tool error (division by zero) and confirm the loop
#    reports it back as a tool result instead of crashing the program.
# ---------------------------------------------------------------------------
def exercise_tool_error_handling() -> None:
    print("\n=== Exercise: a tool that raises an error ===")
    q = "What is 10 divided by 0?"
    try:
        answer = run_tool_calling_loop(q)
        print(f"Q: {q}\nA: {answer}")
        print("(The ValueError from safe_divide was caught inside the loop and reported")
        print(" back as the tool's result -- the script kept running instead of crashing.)")
    except Exception as exc:  # pragma: no cover - this should NOT happen if the loop's
        # try/except around selected_tool.invoke() is working correctly.
        print(f"Unhandled error reached the exercise -- the loop's error handling failed: {exc}")


if __name__ == "__main__":
    try:
        run_demo()
        exercise_tool_error_handling()
    except Exception as exc:  # top-level guard so the script fails loudly, not silently
        print(f"Unexpected error while running the tool-calling demo: {exc}")
        sys.exit(1)
