"""
Statute retrieval support for Nigerian Land Use Act and Evidence Act sections.
"""
from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


logger = logging.getLogger(__name__)


SECTION_HEADING_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:section|s\.)\s*(\d+[a-zA-Z]?)\s*[\.\-:\)]?\s*([^\n]{0,140})\s*$"
    ),
    re.compile(
        r"(?im)^\s*(\d+[a-zA-Z]?)\s*[\.\-:\)]\s*([A-Z][^\n]{3,140})\s*$"
    ),
]
EXPLICIT_SECTION_REF_PATTERN = re.compile(r"(?i)\bsection\s+(\d+[a-zA-Z]?)\b")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _section_numeric_value(section_number: str) -> int:
    match = re.match(r"(\d+)", str(section_number))
    return int(match.group(1)) if match else 10**9


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _infer_statute_name(source: str) -> str:
    source_lower = source.lower()
    if "land" in source_lower and "use" in source_lower and "act" in source_lower:
        return "Land Use Act 2004"
    if "evidence" in source_lower and "act" in source_lower:
        return "Evidence Act 2011"

    stem = Path(source).stem if not source.startswith("http") else source
    stem = stem.replace("-", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in stem.split()) or "Unknown Statute"


class StatuteKnowledgeBase:
    """
    Loads statute text sources, parses sections, and ranks relevant sections for a query.
    """

    def __init__(
        self,
        local_pdf_paths: Optional[List[str | Path]] = None,
        online_urls: Optional[List[str]] = None,
    ):
        self.local_pdf_paths = [Path(p) for p in (local_pdf_paths or []) if str(p).strip()]
        self.online_urls = [u.strip() for u in (online_urls or []) if u and u.strip()]

        self.sections: List[Dict[str, Any]] = []
        self.section_vectorizer: Optional[TfidfVectorizer] = None
        self.section_matrix = None
        self.loaded_sources: List[str] = []
        self._last_error: Optional[str] = None

    def _read_pdf(self, pdf_path: Path) -> str:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Statute PDF not found: {pdf_path}")

        text_chunks: List[str] = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text:
                        text_chunks.append(text)
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    text_chunks.append(text)

        return _normalize_whitespace("\n".join(text_chunks))

    def _read_online_text(self, url: str) -> str:
        req = urllib.request.Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                raw = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch statute URL {url}: {exc}") from exc

        text = raw.decode("utf-8", errors="ignore")
        if "html" in content_type or "<html" in text.lower():
            text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
            text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
        return _normalize_whitespace(text)

    def _parse_sections(self, text: str, source: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        cleaned = _normalize_whitespace(text)
        statute_name = _infer_statute_name(source)
        statute_key = _slugify(statute_name)

        best_matches: List[re.Match] = []
        best_score = -1.0
        for pattern in SECTION_HEADING_PATTERNS:
            candidate_matches = list(pattern.finditer(cleaned))
            if not candidate_matches:
                continue
            numbers = [_section_numeric_value(m.group(1)) for m in candidate_matches]
            sequential_hits = sum(
                1
                for i in range(1, len(numbers))
                if numbers[i] >= numbers[i - 1] and numbers[i] - numbers[i - 1] <= 2
            )
            score = float(len(candidate_matches)) + (0.15 * sequential_hits)
            if score > best_score:
                best_score = score
                best_matches = candidate_matches

        if not best_matches:
            logger.warning("No explicit section headings parsed for source: %s", source)
            return []

        parsed_sections: List[Dict[str, Any]] = []
        for idx, match in enumerate(best_matches):
            sec_num = str(match.group(1)).upper()
            sec_title = (match.group(2) or "").strip(" .:-\t")
            body_start = match.end()
            body_end = best_matches[idx + 1].start() if idx + 1 < len(best_matches) else len(cleaned)
            body = cleaned[body_start:body_end].strip()
            if len(body) < 20:
                continue

            section_id = f"Section {sec_num}"
            parsed_sections.append(
                {
                    "section_uid": f"{statute_key}::{sec_num}",
                    "statute_name": statute_name,
                    "statute_key": statute_key,
                    "section_id": section_id,
                    "section_number": sec_num,
                    "title": sec_title if sec_title else section_id,
                    "text": body,
                    "source": source,
                }
            )

        return parsed_sections

    def _build_index(self) -> None:
        if not self.sections:
            self.section_vectorizer = None
            self.section_matrix = None
            return

        docs = [
            f"{s['statute_name']} {s['section_id']} {s['title']} {s['text']}"
            for s in self.sections
        ]
        self.section_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )
        self.section_matrix = self.section_vectorizer.fit_transform(docs)

    def load(self) -> bool:
        """Load all configured local and online statute sources."""
        self.sections = []
        self.loaded_sources = []
        self._last_error = None

        for pdf_path in self.local_pdf_paths:
            try:
                text = self._read_pdf(pdf_path)
                sections = self._parse_sections(text=text, source=str(pdf_path))
                if sections:
                    self.sections.extend(sections)
                    self.loaded_sources.append(str(pdf_path))
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Failed statute PDF load %s: %s", pdf_path, exc)

        for url in self.online_urls:
            try:
                text = self._read_online_text(url)
                sections = self._parse_sections(text=text, source=url)
                if sections:
                    self.sections.extend(sections)
                    self.loaded_sources.append(url)
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Failed statute URL load %s: %s", url, exc)

        deduped: Dict[str, Dict[str, Any]] = {}
        for section in self.sections:
            section_uid = str(section["section_uid"])
            if section_uid not in deduped:
                deduped[section_uid] = section
                continue
            if len(section.get("text", "")) > len(deduped[section_uid].get("text", "")):
                deduped[section_uid] = section

        self.sections = sorted(
            deduped.values(),
            key=lambda item: (
                item.get("statute_name", ""),
                _section_numeric_value(item.get("section_number", "")),
                item.get("section_number", ""),
            ),
        )
        self._build_index()

        if self.sections:
            logger.info(
                "Loaded statute knowledge base: %s sections from %s sources",
                len(self.sections),
                len(self.loaded_sources),
            )
            return True

        logger.warning("Statute knowledge base unavailable (no sections loaded).")
        return False

    def is_ready(self) -> bool:
        return bool(self.sections and self.section_vectorizer is not None and self.section_matrix is not None)

    def status(self) -> Dict[str, Any]:
        statute_names = sorted({sec.get("statute_name", "") for sec in self.sections if sec.get("statute_name")})
        return {
            "ready": self.is_ready(),
            "sections_count": len(self.sections),
            "sources": self.loaded_sources,
            "statutes_loaded": statute_names,
            "last_error": self._last_error,
        }

    def list_all_sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "statute_name": sec["statute_name"],
                "section_id": sec["section_id"],
                "section_number": sec["section_number"],
                "title": sec["title"],
                "source": sec["source"],
            }
            for sec in self.sections
        ]

    def _score_boost(self, query: str, section: Dict[str, Any]) -> float:
        boost = 0.0
        query_lower = query.lower()
        section_num = section.get("section_number", "").lower()
        statute_name = section.get("statute_name", "").lower()

        explicit_refs = EXPLICIT_SECTION_REF_PATTERN.findall(query_lower)
        if section_num and section_num.lower() in [ref.lower() for ref in explicit_refs]:
            boost += 0.35

        if section.get("section_id", "").lower() in query_lower:
            boost += 0.25

        if "land use act" in query_lower and "land use act" in statute_name:
            boost += 0.18
        if "evidence act" in query_lower and "evidence act" in statute_name:
            boost += 0.18

        return boost

    def rank_sections(self, query: str) -> List[Dict[str, Any]]:
        if not self.is_ready() or not query or not query.strip():
            return []

        q = query.strip()
        query_vec = self.section_vectorizer.transform([q])
        similarity_scores = linear_kernel(query_vec, self.section_matrix).flatten()

        ranked: List[Dict[str, Any]] = []
        for idx, base_score in enumerate(similarity_scores):
            section = self.sections[idx]
            final_score = float(base_score) + self._score_boost(q, section)
            ranked.append(
                {
                    "statute_name": section["statute_name"],
                    "section_id": section["section_id"],
                    "section_number": section["section_number"],
                    "title": section["title"],
                    "source": section["source"],
                    "score": final_score,
                    "snippet": section["text"][:420],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _build_top_with_statute_coverage(self, ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not ranked:
            return []

        selected: List[Dict[str, Any]] = []
        seen_keys = set()

        by_statute: Dict[str, List[Dict[str, Any]]] = {}
        for row in ranked:
            by_statute.setdefault(row.get("statute_name", "Unknown Statute"), []).append(row)

        for statute_name in sorted(by_statute.keys()):
            if by_statute[statute_name]:
                top_item = by_statute[statute_name][0]
                uid = f"{top_item.get('statute_name')}::{top_item.get('section_number')}"
                if uid not in seen_keys:
                    seen_keys.add(uid)
                    selected.append(top_item)

        for row in ranked:
            if len(selected) >= top_k:
                break
            uid = f"{row.get('statute_name')}::{row.get('section_number')}"
            if uid in seen_keys:
                continue
            seen_keys.add(uid)
            selected.append(row)

        return selected[: max(1, top_k)]

    def get_related_sections(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.02,
        max_all: int = 80,
    ) -> Dict[str, List[Dict[str, Any]]]:
        ranked = self.rank_sections(query=query)
        if not ranked:
            return {"top": [], "all": [], "per_statute_top": []}

        filtered = [item for item in ranked if item["score"] >= min_score]
        if not filtered:
            filtered = ranked[: max(top_k, 5)]

        # Ensure statute coverage (e.g., Land Use Act + Evidence Act) even when one act scores lower.
        statutes_in_filtered = {item.get("statute_name", "") for item in filtered}
        statutes_in_ranked = {item.get("statute_name", "") for item in ranked}
        missing_statutes = statutes_in_ranked - statutes_in_filtered
        if missing_statutes:
            for statute_name in sorted(missing_statutes):
                fallback = next(
                    (row for row in ranked if row.get("statute_name", "") == statute_name),
                    None,
                )
                if fallback is not None:
                    filtered.append(fallback)
            filtered.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        all_related = filtered[:max_all]
        top_related = self._build_top_with_statute_coverage(all_related, top_k)
        per_statute_top = self._build_top_with_statute_coverage(all_related, top_k=25)
        return {"top": top_related, "all": all_related, "per_statute_top": per_statute_top}
