import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .schemas import ANSWER_SCHEMA, JUDGE_SCHEMA


load_dotenv()


MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
USE_MOCK = os.getenv("USE_MOCK_LLM", "0") == "1"


def _client() -> OpenAI:
    return OpenAI()


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    after = text.split(start, 1)[1]
    if end in after:
        return after.split(end, 1)[0].strip()
    return after.strip()


def answer_with_llm(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any] = ANSWER_SCHEMA,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    model = model or MODEL_DEFAULT
    start = time.time()

    if USE_MOCK:
        joined = "\n".join(m["content"] for m in messages)
        if "EVIDENCE:" not in joined:
            result = {
                "final_answer": "Mock answer generated without evidence.",
                "cited_chunks": [],
                "refused": False,
            }
        else:
            evidence = _extract_between(joined, "EVIDENCE:", "\n\nQUESTION:")
            _ = _extract_between(joined, "QUESTION:", "\n\nINSTRUCTIONS:")
            if not evidence.strip():
                result = {
                    "final_answer": (
                        "Refused: no supporting evidence was retrieved. "
                        "Please provide sources that state the missing details."
                    ),
                    "cited_chunks": [],
                    "refused": True,
                }
            else:
                first_line = evidence.splitlines()[0].strip()
                result = {
                    "final_answer": f"{first_line}",
                    "cited_chunks": [],
                    "refused": False,
                }
        latency = (time.time() - start) * 1000.0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return result, usage, latency

    client = _client()
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=0.2,
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_schema",
                "schema": schema,
                "strict": True,
            }
        },
    )
    output_text = response.output_text
    result = json.loads(output_text)
    usage = response.usage.model_dump() if response.usage else {}
    latency = (time.time() - start) * 1000.0
    return result, usage, latency


def judge_with_llm(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any] = JUDGE_SCHEMA,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    model = model or MODEL_DEFAULT
    start = time.time()

    if USE_MOCK:
        joined = "\n".join(m["content"] for m in messages)
        ground_truth = _extract_between(joined, "GROUND_TRUTH:", "\n\nANSWER:")
        answer = _extract_between(joined, "ANSWER:", "\n\nCITATIONS:")
        is_correct = ground_truth.lower() in answer.lower() if ground_truth else False
        has_unsupported = not is_correct and bool(answer.strip())
        result = {
            "is_correct": is_correct,
            "has_unsupported_claims": has_unsupported,
            "refusal_appropriate": False,
            "notes": "mock-judge",
        }
        latency = (time.time() - start) * 1000.0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return result, usage, latency

    client = _client()
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=0.0,
        text={
            "format": {
                "type": "json_schema",
                "name": "judge_schema",
                "schema": schema,
                "strict": True,
            }
        },
    )
    output_text = response.output_text
    result = json.loads(output_text)
    usage = response.usage.model_dump() if response.usage else {}
    latency = (time.time() - start) * 1000.0
    return result, usage, latency


def estimate_cost(usage: Dict[str, Any], model: str) -> float:
    price_table = {
        "gpt-4.1-mini": (0.0003, 0.0012),
        "gpt-4.1": (0.003, 0.012),
    }
    if not usage:
        return 0.0
    input_price, output_price = price_table.get(model, price_table["gpt-4.1-mini"])
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return (prompt_tokens / 1000.0) * input_price + (completion_tokens / 1000.0) * output_price
