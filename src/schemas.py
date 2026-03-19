ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "cited_chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "offset": {"type": "string"},
                },
                "required": ["quote", "doc_id", "offset"],
                "additionalProperties": False,
            },
        },
        "refused": {"type": "boolean"},
    },
    "required": ["final_answer", "cited_chunks", "refused"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "has_unsupported_claims": {"type": "boolean"},
        "refusal_appropriate": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "is_correct",
        "has_unsupported_claims",
        "refusal_appropriate",
        "notes",
    ],
    "additionalProperties": False,
}
