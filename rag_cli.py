import argparse
import json
import os

from src.pipeline import keyword_baseline, naive_gpt_baseline, rag_with_refusal
from src.vector_store import VectorStore


def cmd_ingest(args: argparse.Namespace) -> None:
    VectorStore.build_from_dir(args.data_dir, args.store_dir)
    print(f"Vector store written to {args.store_dir}")


def cmd_answer(args: argparse.Namespace) -> None:
    store = VectorStore.load(args.store_dir)
    question = args.question
    model = args.model

    if args.method == "keyword":
        result, _, _ = keyword_baseline(question, store, model)
    elif args.method == "naive_gpt":
        result, _, _ = naive_gpt_baseline(question, model)
    else:
        result, _, _, _ = rag_with_refusal(question, store, model)

    print(json.dumps(result, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    os.system(f"python evaluate.py --results {args.results} --questions {args.questions}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--data_dir", default="data")
    ingest.add_argument("--store_dir", default="data/vector_store")
    ingest.set_defaults(func=cmd_ingest)

    answer = sub.add_parser("answer")
    answer.add_argument("--store_dir", default="data/vector_store")
    answer.add_argument("--method", choices=["keyword", "naive_gpt", "rag"], default="rag")
    answer.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    answer.add_argument("--question", required=True)
    answer.set_defaults(func=cmd_answer)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--results", default="results/results.csv")
    evaluate.add_argument("--questions", default="data/questions.jsonl")
    evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
