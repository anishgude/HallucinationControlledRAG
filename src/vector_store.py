import json
import os
from dataclasses import dataclass
from typing import List, Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    text: str
    start_line: int
    end_line: int


class VectorStore:
    def __init__(self, vectorizer: TfidfVectorizer, matrix: np.ndarray, chunks: List[Chunk]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @staticmethod
    def _chunk_markdown(doc_id: str, text: str) -> List[Chunk]:
        lines = text.splitlines()
        chunks: List[Chunk] = []
        current: List[str] = []
        start_line = 1

        def flush(end_line: int) -> None:
            if current:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append(Chunk(doc_id, chunk_text, start_line, end_line))

        for idx, line in enumerate(lines, start=1):
            if line.strip() == "":
                flush(idx - 1)
                current = []
                start_line = idx + 1
            else:
                current.append(line)
        flush(len(lines))
        return chunks

    @classmethod
    def build_from_dir(cls, data_dir: str, out_dir: str) -> "VectorStore":
        chunks: List[Chunk] = []
        for name in sorted(os.listdir(data_dir)):
            if not (name.endswith(".md") or name.endswith(".txt")):
                continue
            doc_id = os.path.splitext(name)[0]
            path = os.path.join(data_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks.extend(cls._chunk_markdown(doc_id, text))

        texts = [c.text for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)

        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
        joblib.dump(matrix, os.path.join(out_dir, "matrix.joblib"))

        with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in chunks:
                record = {
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                f.write(json.dumps(record) + "\n")

        return cls(vectorizer, matrix, chunks)

    @classmethod
    def load(cls, store_dir: str) -> "VectorStore":
        vectorizer = joblib.load(os.path.join(store_dir, "vectorizer.joblib"))
        matrix = joblib.load(os.path.join(store_dir, "matrix.joblib"))
        chunks: List[Chunk] = []
        with open(os.path.join(store_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                chunks.append(
                    Chunk(
                        record["doc_id"],
                        record["text"],
                        record["start_line"],
                        record["end_line"],
                    )
                )
        return cls(vectorizer, matrix, chunks)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "score": float(scores[idx]),
                    "offset": f"lines {chunk.start_line}-{chunk.end_line}",
                }
            )
        return results
