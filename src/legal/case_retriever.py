"""
Case retrieval index for citing similar trained cases during prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


class CaseReferenceIndex:
    """Build/search TF-IDF index over trained-case texts."""

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self.records: List[Dict[str, Any]] = records or []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._build()

    def _build(self) -> None:
        valid_records = []
        docs = []
        for row in self.records:
            text = _clean_text(row.get("retrieval_text") or row.get("full_text") or "")
            if not text:
                continue
            valid_records.append(row)
            docs.append(text)

        self.records = valid_records
        if not docs:
            self.vectorizer = None
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            strip_accents="unicode",
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def is_ready(self) -> bool:
        return bool(self.records and self.vectorizer is not None and self.matrix is not None)

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "records_count": len(self.records),
        }

    def search(self, query: str, top_k: int = 5, min_score: float = 0.03) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []
        q = _clean_text(query)
        if not q:
            return []

        query_vec = self.vectorizer.transform([q])
        scores = linear_kernel(query_vec, self.matrix).flatten()

        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            if len(results) >= max(1, top_k):
                break
            score_val = float(score)
            if score_val < min_score and results:
                continue
            record = self.records[idx]
            snippet_source = _clean_text(
                record.get("facts")
                or record.get("decision")
                or record.get("retrieval_text")
                or record.get("full_text")
            )
            results.append(
                {
                    "case_id": record.get("case_id"),
                    "case_title": record.get("case_title") or "Unnamed Case",
                    "outcome": record.get("outcome"),
                    "court": record.get("court"),
                    "year": record.get("year"),
                    "source_file": record.get("source_file"),
                    "similarity": score_val,
                    "snippet": snippet_source[:420],
                }
            )

        return results
