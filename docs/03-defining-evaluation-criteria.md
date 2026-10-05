# Define Your Evaluation Criteria

**Why:** Before you measure anything, you have to decide what you're measuring. Most weak evals fail right here. They score something vague.

**The idea:** “Good” is too fuzzy to score, so break it down into smaller things you can actually check. A criterion works if two people would look at the same answer and agree on the grade.

For example:

> “Answers well” is too vague.\
>  “Contains the 15-day figure from the handbook” is something you can check.

| Criterion    | Weak version       | Testable version                                      | Scorer type  |
| ------------ | ------------------ | ----------------------------------------------------- | ------------ |
| Correctness  | “Answers well”     | Contains the key fact from the handbook               | Code + judge |
| Faithfulness | “Doesn't lie”      | Makes no claims absent from the context               | Judge        |
| Abstention   | “Knows its limits” | Says “I don't know” when the answer isn't in the docs | Code         |
| Retrieval    | “Finds stuff”      | Retrieved chunks contain the answer                   | Code         |
| Speed        | “Fast enough”      | Responds in under 5 seconds                           | Code         |

**Build:** Create a `CRITERIA.md` file with 5–8 criteria for your RAG agent, using the same table format. For each one, add:

- How it will be scored: code, judge, or both
- One real example that passes
- One real example that fails

For example, for **abstention**:

- **Pass:** The agent says “I don't know” when the docs don't contain the answer.
- **Fail:** The agent makes up an answer.

  **Done when:** For every criterion, you can answer:

- How is it scored?
- What does a passing answer look like?
- What does a failing answer look like?

If you can't answer those clearly, the criterion is probably still too vague.
