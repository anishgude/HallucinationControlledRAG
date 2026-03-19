import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.llm import judge_with_llm, estimate_cost
from src.schemas import JUDGE_SCHEMA


def load_questions(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return {record["id"]: record for record in (json.loads(line) for line in f)}


def judge_row(row: Dict, question: Dict, model: str) -> Dict:
    citations = row.get("cited_chunks", "")
    messages = [
        {
            "role": "system",
            "content": "You are a strict grader. Use only the provided ground truth and citations.",
        },
        {
            "role": "user",
            "content": (
                "QUESTION:\n"
                f"{question['question']}\n\n"
                "GROUND_TRUTH:\n"
                f"{question['ground_truth']}\n\n"
                "ANSWER:\n"
                f"{row['final_answer']}\n\n"
                "CITATIONS:\n"
                f"{citations}\n\n"
                "INSTRUCTIONS:\n"
                "Return JSON with is_correct, has_unsupported_claims, refusal_appropriate, notes."
            ),
        },
    ]
    result, usage, latency = judge_with_llm(messages, JUDGE_SCHEMA, model=model)
    result["judge_latency_ms"] = round(latency, 2)
    result["judge_cost_usd"] = round(estimate_cost(usage, model), 6)
    if not question["answerable"]:
        result["refusal_appropriate"] = bool(row.get("refused"))
    return result


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        answerable = sub[sub["answerable"] == True]
        unanswerable = sub[sub["answerable"] == False]

        accuracy = answerable["is_correct"].mean() if not answerable.empty else 0.0
        unsupported_rate = sub["has_unsupported_claims"].mean() if not sub.empty else 0.0

        true_refusals = unanswerable[unanswerable["refused"] == True]
        total_refusals = sub[sub["refused"] == True]
        refusal_precision = (
            len(true_refusals) / len(total_refusals) if len(total_refusals) > 0 else 0.0
        )
        refusal_recall = len(true_refusals) / len(unanswerable) if len(unanswerable) > 0 else 0.0

        avg_cost = sub["cost_usd"].mean() + sub["judge_cost_usd"].mean()
        avg_latency = sub["latency_ms"].mean() + sub["judge_latency_ms"].mean()

        rows.append(
            {
                "method": method,
                "accuracy_on_answerable": round(accuracy, 3),
                "unsupported_claim_rate": round(unsupported_rate, 3),
                "refusal_precision": round(refusal_precision, 3),
                "refusal_recall": round(refusal_recall, 3),
                "avg_cost_usd": round(avg_cost, 6),
                "avg_latency_ms": round(avg_latency, 2),
            }
        )

    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, out_path: str) -> None:
    lines = []
    lines.append("# RAG Hallucination Study Report")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append(metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![metrics](metrics.png)")
    lines.append("")
    lines.append("![cost_latency](cost_latency.png)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--report", default="results/report.md")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    df = pd.read_csv(args.results)

    judge_records: List[Dict] = []
    for _, row in df.iterrows():
        question = questions[row["question_id"]]
        judge = judge_row(row.to_dict(), question, args.model)
        judge_records.append(judge)

    judge_df = pd.DataFrame(judge_records)
    merged = pd.concat([df.reset_index(drop=True), judge_df], axis=1)
    merged.loc[(merged["answerable"] == False) & (merged["refused"] == True), "has_unsupported_claims"] = False

    metrics = compute_metrics(merged)
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 4))
    metrics.set_index("method")[["accuracy_on_answerable", "unsupported_claim_rate"]].plot(
        kind="bar", ylim=(0, 1)
    )
    plt.title("Accuracy vs Unsupported Claim Rate")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig("results/metrics.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    metrics.set_index("method")[["avg_cost_usd", "avg_latency_ms"]].plot(kind="bar")
    plt.title("Average Cost and Latency")
    plt.tight_layout()
    plt.savefig("results/cost_latency.png")
    plt.close()

    write_report(metrics, args.report)
    merged.to_csv(args.results, index=False)
    print(f"Updated results with judge columns and wrote report to {args.report}")


if __name__ == "__main__":
    main()
