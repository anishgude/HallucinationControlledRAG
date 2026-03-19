import argparse
import csv
import json
import os
from typing import Dict, List

from src.llm import estimate_cost
from src.pipeline import keyword_baseline, naive_gpt_baseline, rag_with_refusal
from src.vector_store import VectorStore


def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--store_dir", default="data/vector_store")
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--output", default="results/results.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    store = VectorStore.build_from_dir(args.data_dir, args.store_dir)
    questions = load_questions(args.questions)

    rows = []
    for item in questions:
        qid = item["id"]
        question = item["question"]
        answerable = item["answerable"]
        ground_truth = item["ground_truth"]

        result, usage, latency = keyword_baseline(question, store, args.model)
        rows.append(
            {
                "question_id": qid,
                "method": "keyword",
                "answerable": answerable,
                "ground_truth": ground_truth,
                "final_answer": result["final_answer"],
                "refused": result["refused"],
                "cited_chunks": json.dumps(result["cited_chunks"]),
                "latency_ms": round(latency, 2),
                "cost_usd": round(estimate_cost(usage, args.model), 6),
            }
        )

        result, usage, latency = naive_gpt_baseline(question, args.model)
        rows.append(
            {
                "question_id": qid,
                "method": "naive_gpt",
                "answerable": answerable,
                "ground_truth": ground_truth,
                "final_answer": result["final_answer"],
                "refused": result["refused"],
                "cited_chunks": json.dumps(result["cited_chunks"]),
                "latency_ms": round(latency, 2),
                "cost_usd": round(estimate_cost(usage, args.model), 6),
            }
        )

        result, usage, latency, _ = rag_with_refusal(question, store, args.model)
        rows.append(
            {
                "question_id": qid,
                "method": "rag",
                "answerable": answerable,
                "ground_truth": ground_truth,
                "final_answer": result["final_answer"],
                "refused": result["refused"],
                "cited_chunks": json.dumps(result["cited_chunks"]),
                "latency_ms": round(latency, 2),
                "cost_usd": round(estimate_cost(usage, args.model), 6),
            }
        )

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "method",
                "answerable",
                "ground_truth",
                "final_answer",
                "refused",
                "cited_chunks",
                "latency_ms",
                "cost_usd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
