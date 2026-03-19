import os

os.environ["USE_MOCK_LLM"] = "1"

from jsonschema import validate

from src.pipeline import keyword_baseline, naive_gpt_baseline, rag_with_refusal
from src.schemas import ANSWER_SCHEMA
from src.vector_store import VectorStore


def test_answer_schema_validity(tmp_path):
    store = VectorStore.build_from_dir("data", tmp_path / "store")
    result, _, _ = keyword_baseline("What is the default admin IP?", store, "gpt-4.1-mini")
    validate(instance=result, schema=ANSWER_SCHEMA)

    result, _, _ = naive_gpt_baseline("What is the default admin IP?", "gpt-4.1-mini")
    validate(instance=result, schema=ANSWER_SCHEMA)

    result, _, _, _ = rag_with_refusal("What is the default admin IP?", store, "gpt-4.1-mini")
    validate(instance=result, schema=ANSWER_SCHEMA)


def test_refusal_behavior(tmp_path):
    store = VectorStore.build_from_dir("data", tmp_path / "store")
    result, _, _, _ = rag_with_refusal(
        "What is the warranty period for the AX-200?", store, "gpt-4.1-mini", min_score=0.9
    )
    assert result["refused"] is True
    assert "Refused" in result["final_answer"]
