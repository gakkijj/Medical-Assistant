"""Dependency-free hybrid retrieval for CI and production fallback.

The Milvus path remains available for neural embeddings. This module adds an
auditable BM25 + hashed character n-gram pipeline so retrieval can be tested
without downloading models or using an API key.
"""
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
HEADING_PATTERN = re.compile(r"^(?:[一二三四五六七八九十]+、|\d+[.、]|【.+】)")
PROMPT_INJECTION_TERMS = (
    "忽略之前", "忽略以上", "system prompt", "系统提示词", "扮演开发者",
    "不要遵循", "泄露密钥", "输出api key",
)


def tokenize(text: str) -> List[str]:
    """Return mixed word, Chinese unigram and Chinese bigram tokens."""
    base = TOKEN_PATTERN.findall(text.lower())
    chinese = [token for token in base if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = ["".join(chinese[index:index + 2]) for index in range(len(chinese) - 1)]
    return base + bigrams


def _document_metadata(path: Path) -> Dict[str, str]:
    number = int(path.stem.split("_", 1)[0])
    if number < 10:
        doc_type, source = "lifestyle", "项目演示生活方式资料"
    elif number < 20:
        doc_type, source = "disease_classification", "项目演示疾病分类资料"
    elif number < 30:
        doc_type, source = "clinical_guideline", "项目演示临床指南资料"
    else:
        doc_type, source = "general", "项目演示医学资料"
    return {
        "doc_id": f"{doc_type}_{path.stem}",
        "type": doc_type,
        "source": source,
        "filename": path.name,
    }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    content: str
    source: str
    doc_type: str
    trusted: bool

    def citation(self, score: float) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "score": round(score, 4),
        }


def load_chunks(document_dir: Path, max_chars: int = 700) -> List[Chunk]:
    """Load text documents and split on sections before applying a size cap."""
    chunks: List[Chunk] = []
    for path in sorted(document_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        lines = [line.rstrip() for line in text.splitlines()]
        if not lines:
            continue
        metadata = _document_metadata(path)
        title = next((line.strip() for line in lines if line.strip()), path.stem)
        section = title
        buffer: List[str] = []
        section_index = 0

        def flush() -> None:
            nonlocal buffer, section_index
            content = "\n".join(buffer).strip()
            if not content:
                return
            pieces = [content[start:start + max_chars] for start in range(0, len(content), max_chars)]
            for piece_index, piece in enumerate(pieces):
                unsafe = any(term in piece.lower() for term in PROMPT_INJECTION_TERMS)
                chunks.append(Chunk(
                    chunk_id=f"{metadata['doc_id']}:{section_index}:{piece_index}",
                    doc_id=metadata["doc_id"],
                    title=title,
                    section=section,
                    content=piece,
                    source=metadata["source"],
                    doc_type=metadata["type"],
                    trusted=not unsafe,
                ))
            section_index += 1
            buffer = []

        for line in lines[1:]:
            stripped = line.strip()
            if stripped and HEADING_PATTERN.match(stripped) and buffer:
                flush()
                section = stripped
            buffer.append(line)
        flush()
    return chunks


class HybridRetriever:
    """BM25 + hashed n-gram retrieval with deterministic reranking."""

    def __init__(self, chunks: Sequence[Chunk], vector_dimensions: int = 384):
        if not chunks:
            raise ValueError("at least one chunk is required")
        self.chunks = list(chunks)
        self.vector_dimensions = vector_dimensions
        self.tokens = [tokenize(f"{chunk.title} {chunk.section} {chunk.content}") for chunk in chunks]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter = Counter()
        for terms in self.term_frequencies:
            self.document_frequency.update(terms.keys())
        self.average_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.vectors = [self._hash_vector(tokens) for tokens in self.tokens]

    @classmethod
    def from_directory(cls, document_dir: Path) -> "HybridRetriever":
        return cls(load_chunks(document_dir))

    def _hash_vector(self, tokens: Iterable[str]) -> Dict[int, float]:
        vector: Dict[int, float] = defaultdict(float)
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.vector_dimensions
            vector[index] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {index: value / norm for index, value in vector.items()}

    @staticmethod
    def _cosine(left: Mapping[int, float], right: Mapping[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def _bm25(self, query_tokens: Sequence[str], index: int) -> float:
        score = 0.0
        frequencies = self.term_frequencies[index]
        length = len(self.tokens[index])
        k1, b = 1.5, 0.75
        for term in set(query_tokens):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self.document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(self.chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * length / self.average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _rank(scores: Sequence[float]) -> Dict[int, int]:
        order = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
        return {index: rank for rank, index in enumerate(order, 1)}

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        if strategy not in {"bm25", "vector", "hybrid"}:
            raise ValueError("strategy must be one of: bm25, vector, hybrid")
        query_tokens = tokenize(query)
        query_vector = self._hash_vector(query_tokens)
        bm25_scores = [self._bm25(query_tokens, index) for index in range(len(self.chunks))]
        vector_scores = [self._cosine(query_vector, vector) for vector in self.vectors]
        bm25_rank = self._rank(bm25_scores)
        vector_rank = self._rank(vector_scores)
        query_term_set = set(query_tokens)

        scored: List[Tuple[float, int]] = []
        for index, chunk in enumerate(self.chunks):
            if filter_type and chunk.doc_type != filter_type:
                continue
            if not chunk.trusted:
                continue
            rank_score = {
                "bm25": 1 / (60 + bm25_rank[index]),
                "vector": 1 / (60 + vector_rank[index]),
                # The small Chinese corpus benefits from a lexical-heavy fusion;
                # the vector branch mainly recovers paraphrases and tie-breaks.
                "hybrid": 2 / (60 + bm25_rank[index]) + 1 / (60 + vector_rank[index]),
            }[strategy]
            title_tokens = set(tokenize(f"{chunk.title} {chunk.section}"))
            title_coverage = len(query_term_set & title_tokens) / max(len(query_term_set), 1)
            exact_bonus = 0.05 if query.strip() and query.strip() in chunk.content else 0.0
            score = rank_score + 0.02 * title_coverage + exact_bonus
            scored.append((score, index))

        results: List[Dict[str, Any]] = []
        seen_documents = set()
        for score, index in sorted(scored, reverse=True):
            chunk = self.chunks[index]
            if chunk.doc_id in seen_documents:
                continue
            seen_documents.add(chunk.doc_id)
            results.append({
                "id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "source": chunk.source,
                    "type": chunk.doc_type,
                    "trusted": chunk.trusted,
                },
                "score": round(score, 6),
                "citation": chunk.citation(score),
            })
            if len(results) >= top_k:
                break
        return results
