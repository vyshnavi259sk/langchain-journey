# Daily Journal

## 2026-11-15
Day 1 of daily commits. Continuing to build consistency.

## 2026-11-14
**Day 2: LCEL Runnables — RunnableParallel, RunnableLambda, and RunnablePassthrough**

Day 8 already chained two prompts with the `|` operator, which is LCEL's `RunnableSequence` under the hood. Today's focus is the rest of the Runnable interface that makes LCEL composable beyond a straight line:

- **RunnablePassthrough** — forwards an input (or a sub-key of it) unchanged into a later step, so the original user question can still reach the final prompt even after an intermediate step (like a retriever) has transformed the data.
- **RunnableParallel** — runs multiple branches on the same input at once and merges their outputs into a dict, e.g. `RunnableParallel(context=retriever, question=RunnablePassthrough())`. This is the pattern that later powers RAG chains, where "context" and "question" need to reach the prompt together.
- **RunnableLambda** — wraps a plain Python function as a Runnable so custom logic (formatting, filtering, light post-processing) can sit inside a `|` chain without breaking the interface.

Practical exercise: build a chain that takes a topic string, fans it out with `RunnableParallel` into two branches — one that passes the topic through unchanged and one that wraps a small Python function (`RunnableLambda`) to uppercase it — then merges both into a prompt template before sending to the LLM. This mirrors the shape of a real retrieval chain (`{"context": retriever, "question": RunnablePassthrough()}`) without needing a vector store yet, which is the next topic in the roadmap (document loaders → embeddings → vector stores → retrieval).
