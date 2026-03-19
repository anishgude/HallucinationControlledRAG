import json
from typing import Any, Dict, List, Tuple

from .llm import answer_with_llm
from .vector_store import VectorStore
from .schemas import ANSWER_SCHEMA


def _format_evidence(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"- ({c['doc_id']} | {c['offset']}) {c['text']}")
    return "\n".join(lines)


def keyword_baseline(question: str, store: VectorStore, model: str) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    chunks = store.search(question, top_k=1)
    evidence = _format_evidence(chunks)
    messages = [
        {
            "role": "system",
            "content": "You answer based only on provided evidence.",
        },
        {
            "role": "user",
            "content": (
                "EVIDENCE:\n"
                f"{evidence}\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "INSTRUCTIONS:\n"
                "Return JSON with final_answer, cited_chunks (empty array), and refused (false)."
            ),
        },
    ]
    result, usage, latency = answer_with_llm(messages, ANSWER_SCHEMA, model=model)
    result["cited_chunks"] = []
    result["refused"] = False
    return result, usage, latency


def naive_gpt_baseline(question: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    messages = [
        {
            "role": "system",
            "content": "You answer the question directly.",
        },
        {
            "role": "user",
            "content": (
                "QUESTION:\n"
                f"{question}\n\n"
                "INSTRUCTIONS:\n"
                "Return JSON with final_answer, cited_chunks (empty array), and refused (false)."
            ),
        },
    ]
    result, usage, latency = answer_with_llm(messages, ANSWER_SCHEMA, model=model)
    result["cited_chunks"] = []
    result["refused"] = False
    return result, usage, latency


def rag_with_refusal(
    question: str,
    store: VectorStore,
    model: str,
    top_k: int = 4,
    min_score: float = 0.15,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[Dict[str, Any]]]:
    chunks = store.search(question, top_k=top_k)
    sufficient = any(c["score"] >= min_score for c in chunks)
    evidence = _format_evidence(chunks)
    if not sufficient:
        messages = [
            {
                "role": "system",
                "content": "You must refuse when evidence is insufficient.",
            },
            {
                "role": "user",
                "content": (
                    "EVIDENCE:\n\n"
                    "QUESTION:\n"
                    f"{question}\n\n"
                    "INSTRUCTIONS:\n"
                    "Return JSON with final_answer explaining missing evidence, cited_chunks empty, refused true."
                ),
            },
        ]
        result, usage, latency = answer_with_llm(messages, ANSWER_SCHEMA, model=model)
        result["cited_chunks"] = []
        result["refused"] = True
        return result, usage, latency, []

    messages = [
        {
            "role": "system",
            "content": (
                "You answer only using the evidence. Provide citations for every claim. "
                "If evidence is insufficient, refuse."
            ),
        },
        {
            "role": "user",
            "content": (
                "EVIDENCE:\n"
                f"{evidence}\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "INSTRUCTIONS:\n"
                "Return JSON with final_answer, cited_chunks (each with quote, doc_id, offset), refused boolean."
            ),
        },
    ]
    result, usage, latency = answer_with_llm(messages, ANSWER_SCHEMA, model=model)

    if not result.get("cited_chunks"):
        result["cited_chunks"] = [
            {
                "quote": c["text"],
                "doc_id": c["doc_id"],
                "offset": c["offset"],
            }
            for c in chunks[:2]
        ]
    return result, usage, latency, chunks
