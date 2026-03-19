# Naive LLM vs Grounded RAG

## Question
What is the parental leave duration in the company policies?

## Naive LLM Failure Mode
A naive model can easily produce a polished but unsupported answer such as:

> Employees receive 12 weeks of parental leave.

The problem is not fluency. The problem is that the repository corpus does not contain that claim anywhere.

## Grounded RAG Behavior
A grounded system should instead respond with a refusal:

```json
{
  "final_answer": "Refused: the available policy documents do not contain parental leave information. A supported answer would require a policy document that explicitly states parental leave duration.",
  "cited_chunks": [],
  "refused": true
}
```

## Why This Example Matters
This comparison illustrates the core design goal of the repository:

- do not answer merely because the question sounds plausible
- only answer when the corpus provides evidence
- treat refusal as a reliability feature
