# Retrieval Design

## Retriever Choice
This repository uses a lightweight lexical retriever built with:

- `TfidfVectorizer`
- cosine similarity

Implementation lives in [`src/vector_store.py`](/c:/Users/ANISH%20PC/Desktop/RAG/rag-hallucination-study/src/vector_store.py).

## Why TF-IDF Here?
The goal of this project is to study grounded generation and refusal behavior, not to maximize retrieval sophistication.

TF-IDF is a reasonable choice for this repository because it is:

- local and reproducible
- easy to inspect
- fast to build
- dependency-light

That keeps the experiment focused on controllable evidence-grounding behavior rather than infrastructure complexity.

## Chunking Strategy
Documents are split on blank lines. Each chunk is stored with:

- `doc_id`
- `text`
- `start_line`
- `end_line`

Those line offsets are surfaced back into citations so answers can reference the original source span.

## Retrieval Output
Each search result includes:

- document id
- chunk text
- similarity score
- line offset string

The RAG pipeline packages the top-k chunks into the generation prompt as explicit evidence.

## Retrieval Threshold and Refusal
The main reliability control happens in [`src/pipeline.py`](/c:/Users/ANISH%20PC/Desktop/RAG/rag-hallucination-study/src/pipeline.py):

- retrieve top-k chunks
- check whether any chunk exceeds a minimum similarity threshold
- if not, trigger the refusal path

This is a simple but useful control mechanism:

- it prevents answering unsupported questions too confidently
- it makes refusal behavior explicit and testable

## Limitations
This retriever is intentionally simple. It will miss some semantically relevant passages that a dense retriever or hybrid retriever would recover.

Future upgrades that would preserve the overall architecture:

- dense embedding retrieval
- hybrid lexical + semantic ranking
- reranking over retrieved chunks
- chunk overlap or hierarchical chunking
- citation span validation

## Why It Still Demonstrates Good Engineering Judgment
The retrieval layer is small, inspectable, and sufficient for demonstrating:

- the importance of grounding
- the role of evidence thresholds
- the interaction between retrieval quality and refusal behavior
- how evaluation can quantify hallucination reduction
