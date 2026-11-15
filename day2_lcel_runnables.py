"""
Day 2 (journal) - LCEL Runnables: RunnableParallel, RunnableLambda, RunnablePassthrough
------------------------------------------------------------------------------
Companion implementation for the "Day 2: LCEL Runnables" entry in journal.md
(2026-11-14). Named day2_lcel_runnables.py -- distinct from day2.py, which is
an earlier, unrelated JSON-parsing exercise from the day1.py-day8-2.py series;
journal.md's own day count and that file series ended up on separate
numbering tracks, see the note in README.md.
------------------------------------------------------------------------------
Day 8 already chained two prompts with the `|` operator (that is LCEL's
RunnableSequence under the hood). Today's focus is the rest of the Runnable
interface that makes LCEL composable beyond a straight line:

- RunnablePassthrough : forwards an input (or a sub-key of it) unchanged into
  a later step, so the original question can still reach the final prompt
  even after an earlier step (like a retriever) has transformed the data.
- RunnableParallel     : runs multiple branches on the SAME input at once and
  merges their outputs into a dict, e.g.
  RunnableParallel(context=retriever, question=RunnablePassthrough()).
  This is exactly the shape that later powers RAG chains.
- RunnableLambda       : wraps a plain Python function as a Runnable so custom
  logic (formatting, filtering, light post-processing) can sit inside a `|`
  chain without breaking the Runnable interface.

No paid API call is required to understand the composition, so this script
builds the chain with a small local "mock LLM" by default. If a GROQ_API_KEY
is present in the environment, it swaps in a real langchain_groq.ChatGroq
call instead, so the same code demonstrates both the offline-safe path and
the real integration path.
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv()


def shout(text: str) -> str:
    """Plain Python function. RunnableLambda turns this into a Runnable."""
    return text.upper()


def word_count(text: str) -> int:
    """Second plain function used in the exercise at the bottom of the file."""
    return len(text.split())


def build_mock_llm():
    """
    A tiny stand-in for a real chat model, implemented as a RunnableLambda.

    It takes the formatted prompt's text and returns a deterministic string,
    so the whole chain can be run with zero network calls and zero API keys
    -- useful for CI, offline study, and this portfolio repo, where a GROQ_API_KEY
    won't always be configured.
    """

    def _fake_generate(prompt_value) -> str:
        # prompt_value is a StringPromptValue / ChatPromptValue; .to_string()
        # gives the fully rendered prompt text, same as what a real model sees.
        rendered = prompt_value.to_string()
        return f"[mock-llm response] Received a prompt of {len(rendered)} chars."

    return RunnableLambda(_fake_generate)


def build_llm():
    """Use a real ChatGroq model if an API key is configured, else the mock."""
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Falling back to mock LLM (ChatGroq unavailable: {exc})")
    return build_mock_llm()


def main() -> None:
    # 1. RunnableParallel + RunnableLambda + RunnablePassthrough on their own.
    fan_out = RunnableParallel(
        original=RunnablePassthrough(),
        shouted=RunnableLambda(shout),
    )
    result = fan_out.invoke("langchain")
    print("fan_out.invoke('langchain') ->", result)
    assert result == {"original": "langchain", "shouted": "LANGCHAIN"}

    # 2. Feed that fanned-out dict into a prompt template, then into an LLM.
    #    This mirrors the classic RAG shape:
    #    {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    prompt = ChatPromptTemplate.from_template(
        "Topic: {original}\nShouted version: {shouted}\n"
        "Write one short, friendly sentence introducing this topic."
    )
    llm = build_llm()
    chain = fan_out | prompt | llm | StrOutputParser()

    try:
        response = chain.invoke("retrieval augmented generation")
        print("\nchain.invoke('retrieval augmented generation') ->")
        print(response)
    except Exception as exc:
        # Basic error handling: network/credential problems shouldn't crash
        # the whole script, they should surface a clear, actionable message.
        print(f"\nLLM call failed ({exc}); the Runnable composition above "
              "still ran correctly -- only the final generation step needs "
              "a working API key/network connection.")

    # 3. Exercise: extend RunnableParallel with a third, independent branch.
    #    Add `word_count` (defined above) as a third key called "length",
    #    so fan_out.invoke("a longer topic string") also returns
    #    {"original": ..., "shouted": ..., "length": 3}.
    extended_fan_out = RunnableParallel(
        original=RunnablePassthrough(),
        shouted=RunnableLambda(shout),
        length=RunnableLambda(word_count),
    )
    exercise_result = extended_fan_out.invoke("a longer topic string")
    print("\nExercise - extended_fan_out.invoke('a longer topic string') ->")
    print(exercise_result)
    assert exercise_result["length"] == 4


if __name__ == "__main__":
    main()
