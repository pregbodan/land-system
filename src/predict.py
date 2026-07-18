"""
Prediction script and service for new land matter cases.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_collection.text_extraction import TextExtractor
from preprocessing.text_preprocessing import TextPreprocessor, JudgmentParser
from preprocessing.annotation import CaseAnnotator
from preprocessing.structured_data import canonicalize_outcome_label
from legal.statute_retriever import StatuteKnowledgeBase
from legal.case_retriever import CaseReferenceIndex


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LEGAL_FEATURE_COLUMNS = [
    "mentions_land_use_act",
    "has_certificate_of_occupancy",
    "has_governors_consent",
    "mentions_section_22",
    "mentions_section_28",
    "mentions_evidence_act",
    "has_documentary_evidence",
    "has_survey_evidence",
    "mentions_section_84",
    "customary_tenure",
    "statutory_tenure",
    "has_witness_testimony",
    "has_expert_evidence",
    "involves_inheritance",
    "has_boundary_issue",
    "mentions_trespass",
    "involves_fraud",
    "has_oral_evidence",
]
SUCCESS_DECISION_HINTS = [
    "declaration granted",
    "injunction granted",
    "damages awarded",
    "appeal allowed",
    "plaintiff succeeds",
    "entitled to statutory right of occupancy",
    "judgment for the plaintiff",
]
FAILURE_DECISION_HINTS = [
    "appeal dismissed",
    "claim dismissed",
    "relief denied",
    "judgment for the defendant",
    "plaintiff fails",
]
DEFAULT_STATUTE_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data" / "legal" / "Nigerian land use act 2004.pdf"
DEFAULT_LAND_USE_ACT_PATH = Path(__file__).resolve().parent.parent / "data" / "legal" / "Nigerian-land-use-act-2004.pdf"
DEFAULT_EVIDENCE_ACT_PATH = Path(__file__).resolve().parent.parent / "data" / "legal" / "evidence-act-2011.pdf"
OLLAMA_JSON_PATTERN = re.compile(r"\{[\s\S]*\}")
REQUESTER_ROLE_GUIDANCE = {
    "regular_user": (
        "Plain language accepted. Include what happened, who took/owns the land, "
        "whether compensation was paid, and what remedy you want."
    ),
    "lawyer": (
        "Use legal framing with parties, facts, issues, evidence, reliefs sought, and holding."
    ),
    "judge": (
        "Use judicial style with concise facts, issues for determination, analysis, and dispositive orders."
    ),
}


class OfflineOllamaClient:
    """Local-first Ollama client with HTTP + CLI fallback for offline use."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 120,
        allow_cli_fallback: bool = True,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.default_model = default_model or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = max(20, int(timeout))
        self.allow_cli_fallback = bool(allow_cli_fallback)
        self.last_error: Optional[str] = None
        self.last_transport: Optional[str] = None

    def _request(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url=url, data=body, method=method, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8")
                return json.loads(response_text) if response_text else {}
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama API at {self.base_url}: {exc.reason}") from exc

    def list_models(self) -> List[str]:
        response = self._request("GET", "/api/tags")
        models = response.get("models", [])
        return [str(item.get("name", "")).strip() for item in models if str(item.get("name", "")).strip()]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            if self.allow_cli_fallback and shutil.which("ollama"):
                return True
            return False

    def _generate_via_http(self, prompt: str, model: Optional[str]) -> str:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.15},
        }
        response = self._request("POST", "/api/generate", payload=payload)
        text = str(response.get("response", "")).strip()
        if not text:
            raise RuntimeError("Ollama API returned an empty response.")
        self.last_transport = "http"
        return text

    def _generate_via_cli(self, prompt: str, model: Optional[str]) -> str:
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            raise RuntimeError("Ollama CLI not found in PATH.")

        chosen_model = model or self.default_model
        command = [ollama_bin, "run", chosen_model, prompt]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            raise RuntimeError(
                f"Ollama CLI failed (exit={completed.returncode}). {stderr_text}"
            )
        output = (completed.stdout or "").strip()
        if not output:
            raise RuntimeError("Ollama CLI returned an empty response.")
        self.last_transport = "cli"
        return output

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        self.last_error = None
        try:
            return self._generate_via_http(prompt=prompt, model=model)
        except Exception as http_exc:
            if not self.allow_cli_fallback:
                self.last_error = str(http_exc)
                raise
            try:
                return self._generate_via_cli(prompt=prompt, model=model)
            except Exception as cli_exc:
                self.last_error = f"{http_exc} | CLI fallback failed: {cli_exc}"
                raise RuntimeError(self.last_error) from cli_exc

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        content = text.strip()
        if content.startswith("```"):
            lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
            content = "\n".join(lines).strip()
        match = OLLAMA_JSON_PATTERN.search(content)
        if not match:
            raise ValueError("No JSON object found in Ollama response.")
        return json.loads(match.group(0))

    def _normalize_probability_map(
        self,
        raw_probabilities: Dict[str, Any],
        candidate_labels: List[str],
    ) -> Dict[str, float]:
        normalized: Dict[str, float] = {label: 0.0 for label in candidate_labels}
        for key, value in raw_probabilities.items():
            canonical = canonicalize_outcome_label(str(key))
            if canonical in normalized:
                try:
                    normalized[canonical] += float(value)
                except (TypeError, ValueError):
                    continue

        total = float(sum(v for v in normalized.values() if v > 0))
        if total <= 0:
            return {}
        return {label: val / total for label, val in normalized.items()}

    def predict_outcome_probabilities(
        self,
        case_text: str,
        candidate_labels: List[str],
        statute_sections: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, float]:
        clean_labels = [canonicalize_outcome_label(label) for label in candidate_labels]
        clean_labels = [label for label in clean_labels if label]
        clean_labels = sorted(set(clean_labels))
        if not clean_labels:
            clean_labels = ["dismissed", "granted", "matter remitted", "other"]

        statute_lines = []
        for section in (statute_sections or [])[:8]:
            statute_lines.append(
                f"- {section.get('statute_name', 'Statute')} {section.get('section_id', '')}: {section.get('title', '')}"
            )

        prompt = (
            "You are a legal outcome classifier for Nigerian land-matter judgments.\n"
            "Use only these candidate labels and return strict JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "probabilities": {"<label>": <float between 0 and 1>, ...},\n'
            '  "reason": "<one sentence>"\n'
            "}\n"
            f"Allowed labels: {clean_labels}\n"
            "Give higher probability when land ownership, trespass, title, injunction, or damages findings are explicit.\n"
            "Use statute clues when relevant.\n"
            "Relevant statute pointers:\n"
            f"{chr(10).join(statute_lines) if statute_lines else '- none'}\n\n"
            "Case text excerpt:\n"
            f"{case_text[:3000]}\n\n"
            "Return JSON only."
        )

        raw_text = self.generate(prompt=prompt, model=model)
        payload = self._extract_json_object(raw_text)
        raw_probs = payload.get("probabilities", {})
        if not isinstance(raw_probs, dict):
            raise ValueError("Ollama probabilities payload is invalid.")
        return self._normalize_probability_map(raw_probs, clean_labels)


class LandMatterPredictor:
    """Predict outcomes for land matter cases."""

    def __init__(
        self,
        model_path: str | Path,
        artifacts_path: Optional[str | Path] = None,
        preprocessed_data_path: Optional[str | Path] = None,
        training_results_path: Optional[str | Path] = None,
        statute_pdf_paths: Optional[List[str | Path]] = None,
        statute_urls: Optional[List[str]] = None,
        enable_statute_support: bool = True,
        statute_top_k: int = 8,
        top_models_count: int = 3,
        ml_weight: float = 0.8,
        ollama_weight: float = 0.2,
        enable_ollama_hybrid: bool = True,
        ollama_model: Optional[str] = None,
        ollama_timeout: int = 120,
        related_cases_top_k: int = 5,
        feedback_store_path: Optional[str | Path] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model (.pkl)
            artifacts_path: Path to inference_artifacts.pkl or preprocessed data .pkl
            preprocessed_data_path: Backward-compatible alias for artifacts path
            training_results_path: Optional path to training_results.json
            statute_pdf_paths: Optional local statute PDF paths
            statute_urls: Optional statute URLs to fetch text from
            enable_statute_support: Enables statute retrieval augmentation
            statute_top_k: Number of top statute sections to inject into model context
            top_models_count: Number of best ML models to use in the ensemble
            ml_weight: Global weight for ML ensemble probabilities
            ollama_weight: Global weight for Ollama probabilities
            enable_ollama_hybrid: Enables 80/20 ML + Ollama blending
            ollama_model: Preferred Ollama model name
            ollama_timeout: Timeout for Ollama calls in seconds
            related_cases_top_k: Maximum number of related trained cases to return
            feedback_store_path: Path to append feedback cases for incremental learning
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)

        resolved_artifacts_path = self._resolve_artifacts_path(
            explicit_artifacts_path=artifacts_path,
            preprocessed_data_path=preprocessed_data_path,
        )
        logger.info("Loading inference artifacts from %s", resolved_artifacts_path)

        with open(resolved_artifacts_path, "rb") as f:
            artifact_data = pickle.load(f)

        if "feature_engineer" not in artifact_data or "label_encoder" not in artifact_data:
            raise ValueError(
                "Artifacts file must contain 'feature_engineer' and 'label_encoder'."
            )

        self.feature_engineer = artifact_data["feature_engineer"]
        self.label_encoder = artifact_data["label_encoder"]
        self.feature_names = artifact_data.get("feature_names") or getattr(
            self.feature_engineer, "feature_names", []
        )
        self._expected_legal_columns = self._resolve_expected_legal_columns()
        self.training_results_path = self._resolve_training_results_path(training_results_path)
        self.top_models_count = max(1, int(top_models_count))
        self.primary_ml_weight = max(0.0, float(ml_weight))
        self.primary_ollama_weight = max(0.0, float(ollama_weight))
        self.enable_ollama_hybrid = bool(enable_ollama_hybrid)
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3")
        self.ollama_client = OfflineOllamaClient(
            default_model=self.ollama_model,
            timeout=ollama_timeout,
            allow_cli_fallback=True,
        )
        self.ensemble_models = self._load_top_ensemble_models(self.top_models_count)
        self.related_cases_top_k = max(1, int(related_cases_top_k))
        self.feedback_store_path = (
            Path(feedback_store_path)
            if feedback_store_path
            else (self.model_path.parent / "prediction_feedback.jsonl")
        )
        self.case_retriever = self._initialize_case_retriever(artifact_data)

        self.text_extractor = TextExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.judgment_parser = JudgmentParser()
        self.case_annotator = CaseAnnotator()
        self.enable_statute_support = bool(enable_statute_support)
        self.statute_top_k = max(1, int(statute_top_k))
        self.statute_kb = self._initialize_statute_kb(
            statute_pdf_paths=statute_pdf_paths,
            statute_urls=statute_urls,
            enable=self.enable_statute_support,
        )

        logger.info("Predictor initialized successfully")

    def _resolve_expected_legal_columns(self):
        """Infer legal feature columns that were present during training."""
        if not self.feature_names:
            return LEGAL_FEATURE_COLUMNS
        tfidf_feature_count = len(self.feature_engineer.tfidf_vectorizer.get_feature_names_out())
        other_feature_names = self.feature_names[tfidf_feature_count:]
        return [col for col in other_feature_names if col in LEGAL_FEATURE_COLUMNS]

    def _resolve_training_results_path(
        self,
        training_results_path: Optional[str | Path],
    ) -> Optional[Path]:
        """Resolve training results JSON used to pick the best 3 models."""
        candidates: List[Path] = []
        if training_results_path:
            candidates.append(Path(training_results_path))

        model_dir = self.model_path.parent
        project_root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                model_dir / "training_results.json",
                project_root / "data" / "models" / "training_results.json",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _normalize_label(self, label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")

    def _model_key_to_path(self, model_key: str) -> Path:
        return self.model_path.parent / f"{model_key}.pkl"

    def _load_top_ensemble_models(self, top_n: int) -> List[Dict[str, Any]]:
        """
        Load best N models based on training accuracy.
        Falls back to the main model if ranking data is unavailable.
        """
        selected_models: List[Dict[str, Any]] = []
        ranked_candidates: List[Dict[str, Any]] = []

        if self.training_results_path and self.training_results_path.exists():
            try:
                with open(self.training_results_path, "r", encoding="utf-8") as f:
                    results_payload = json.load(f)
                if isinstance(results_payload, dict):
                    for model_name, metrics in results_payload.items():
                        if not isinstance(metrics, dict):
                            continue
                        accuracy = float(metrics.get("accuracy", 0.0))
                        model_key = self._normalize_label(model_name)
                        model_path = self._model_key_to_path(model_key)
                        ranked_candidates.append(
                            {
                                "name": model_name,
                                "model_key": model_key,
                                "model_path": model_path,
                                "accuracy": accuracy,
                            }
                        )
            except Exception as exc:
                logger.warning("Failed to parse training results at %s: %s", self.training_results_path, exc)

        ranked_candidates.sort(key=lambda row: row.get("accuracy", 0.0), reverse=True)

        for candidate in ranked_candidates:
            if len(selected_models) >= top_n:
                break
            model_path = candidate["model_path"]
            if not model_path.exists():
                continue
            try:
                loaded_model = joblib.load(model_path)
                selected_models.append(
                    {
                        "name": candidate["name"],
                        "model_key": candidate["model_key"],
                        "model_path": str(model_path),
                        "model": loaded_model,
                        "accuracy": float(candidate["accuracy"]),
                    }
                )
            except Exception as exc:
                logger.warning("Failed loading ensemble model %s: %s", model_path, exc)

        if not selected_models:
            selected_models.append(
                {
                    "name": "Best Model",
                    "model_key": self.model_path.stem,
                    "model_path": str(self.model_path),
                    "model": self.model,
                    "accuracy": 1.0,
                }
            )

        if all(item.get("accuracy", 0.0) <= 0 for item in selected_models):
            per_model_weight = 1.0 / len(selected_models)
            for item in selected_models:
                item["ensemble_weight"] = per_model_weight
        else:
            total_accuracy = sum(max(0.0, float(item.get("accuracy", 0.0))) for item in selected_models)
            if total_accuracy <= 0:
                total_accuracy = float(len(selected_models))
            for item in selected_models:
                item["ensemble_weight"] = max(0.0, float(item.get("accuracy", 0.0))) / total_accuracy

        logger.info(
            "Loaded %s ensemble model(s): %s",
            len(selected_models),
            [item["name"] for item in selected_models],
        )
        return selected_models

    def _model_probabilities(
        self,
        model_obj: Any,
        X: Any,
    ) -> Dict[str, float]:
        """Get canonical probability map from one model."""
        canonical_map: Dict[str, float] = {}

        if hasattr(model_obj, "predict_proba"):
            probabilities = model_obj.predict_proba(X)[0]
            model_classes = getattr(model_obj, "classes_", None)
            if model_classes is None or len(model_classes) != len(probabilities):
                model_classes = list(range(len(probabilities)))

            for idx, raw_prob in enumerate(probabilities):
                cls_value = model_classes[idx]
                try:
                    class_index = int(cls_value)
                except (TypeError, ValueError):
                    class_index = idx

                if 0 <= class_index < len(self.label_encoder.classes_):
                    raw_label = str(self.label_encoder.classes_[class_index])
                else:
                    raw_label = str(cls_value)
                canonical = canonicalize_outcome_label(raw_label)
                if not canonical:
                    continue
                canonical_map[canonical] = canonical_map.get(canonical, 0.0) + float(raw_prob)
        else:
            prediction_idx = int(model_obj.predict(X)[0])
            raw_label = str(self.label_encoder.inverse_transform([prediction_idx])[0])
            canonical = canonicalize_outcome_label(raw_label)
            if canonical:
                canonical_map[canonical] = 1.0

        total = sum(canonical_map.values())
        if total > 0:
            canonical_map = {label: prob / total for label, prob in canonical_map.items()}
        return canonical_map

    def _merge_probability_maps(
        self,
        weighted_maps: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Merge weighted probability maps into one normalized distribution."""
        merged: Dict[str, float] = {}
        for item in weighted_maps:
            weight = float(item.get("weight", 0.0))
            probs = item.get("probabilities", {})
            if weight <= 0 or not isinstance(probs, dict):
                continue
            for label, value in probs.items():
                merged[label] = merged.get(label, 0.0) + (float(value) * weight)

        total = sum(v for v in merged.values() if v > 0)
        if total <= 0:
            return {}
        return {label: value / total for label, value in merged.items()}

    def _sort_probability_map(self, probs: Dict[str, float]) -> Dict[str, float]:
        return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))

    def _ensemble_probability_map(self, X: Any) -> Dict[str, Any]:
        """Run best-3-model ensemble and return weighted ML probabilities."""
        per_model_outputs: List[Dict[str, Any]] = []
        weighted_maps: List[Dict[str, Any]] = []

        for item in self.ensemble_models:
            model_obj = item["model"]
            model_name = item["name"]
            model_weight = float(item.get("ensemble_weight", 0.0))
            try:
                probabilities = self._model_probabilities(model_obj, X)
            except Exception as exc:
                logger.warning("Model %s prediction failed: %s", model_name, exc)
                continue

            per_model_outputs.append(
                {
                    "model_name": model_name,
                    "model_path": item.get("model_path"),
                    "ensemble_weight": model_weight,
                    "accuracy": item.get("accuracy"),
                    "probabilities": self._sort_probability_map(probabilities),
                }
            )
            weighted_maps.append({"weight": model_weight, "probabilities": probabilities})

        merged = self._merge_probability_maps(weighted_maps)
        return {
            "probabilities": self._sort_probability_map(merged),
            "per_model_outputs": per_model_outputs,
        }

    def _ollama_probability_map(
        self,
        cleaned_text: str,
        candidate_labels: List[str],
        statute_related: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Get Ollama probability map for hybrid mode.
        Returns empty map on failure but preserves error details.
        """
        related_sections = statute_related.get("top", []) if isinstance(statute_related, dict) else []
        try:
            probs = self.ollama_client.predict_outcome_probabilities(
                case_text=cleaned_text,
                candidate_labels=candidate_labels,
                statute_sections=related_sections,
                model=self.ollama_model,
            )
            return {
                "probabilities": self._sort_probability_map(probs),
                "error": None,
                "transport": self.ollama_client.last_transport,
            }
        except Exception as exc:
            return {
                "probabilities": {},
                "error": str(exc),
                "transport": self.ollama_client.last_transport,
            }

    def hybrid_status(self) -> Dict[str, Any]:
        """Expose hybrid config for API/UI."""
        return {
            "top_models_count": len(self.ensemble_models),
            "ml_weight": self.primary_ml_weight,
            "ollama_weight": self.primary_ollama_weight,
            "ollama_hybrid_enabled": self.enable_ollama_hybrid,
            "ollama_model": self.ollama_model,
            "ensemble_models": [
                {
                    "name": item.get("name"),
                    "model_path": item.get("model_path"),
                    "accuracy": item.get("accuracy"),
                    "ensemble_weight": item.get("ensemble_weight"),
                }
                for item in self.ensemble_models
            ],
        }

    def ollama_status(self) -> Dict[str, Any]:
        """Return Ollama availability details."""
        models: List[str] = []
        available = False
        error = None
        cli_available = bool(shutil.which("ollama"))
        try:
            models = self.ollama_client.list_models()
            available = True
        except Exception as exc:
            error = str(exc)
            available = cli_available

        return {
            "available": available,
            "base_url": self.ollama_client.base_url,
            "default_model": self.ollama_client.default_model,
            "models": models,
            "cli_available": cli_available,
            "error": error,
        }

    def _normalize_requester_role(self, requester_role: Optional[str]) -> str:
        role = str(requester_role or "regular_user").strip().lower()
        if role not in REQUESTER_ROLE_GUIDANCE:
            return "regular_user"
        return role

    def requester_role_help(self) -> Dict[str, str]:
        return dict(REQUESTER_ROLE_GUIDANCE)

    def _adapt_input_text_for_role(self, raw_text: str, requester_role: str) -> str:
        """
        Adapt user input style based on requester profile without changing facts.
        Keep transformations minimal to avoid distorting legal signals.
        """
        _ = self._normalize_requester_role(requester_role)
        return str(raw_text or "").strip()

    def _initialize_case_retriever(self, artifact_data: Dict[str, Any]) -> CaseReferenceIndex:
        records = artifact_data.get("case_reference_records", [])
        if not isinstance(records, list):
            records = []
        if not records:
            candidates = [
                self.model_path.parent / "case_reference_records.json",
                Path(__file__).resolve().parent.parent / "data" / "models" / "case_reference_records.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, list):
                        records = loaded
                        break
                except Exception:
                    continue
        retriever = CaseReferenceIndex(records=records)
        if not retriever.is_ready():
            logger.warning(
                "Case reference index is empty. Re-run training to populate case citations in inference artifacts."
            )
        else:
            logger.info("Loaded case reference index with %s records.", len(records))
        return retriever

    def case_retriever_status(self) -> Dict[str, Any]:
        return self.case_retriever.status()

    def get_related_cases(
        self,
        text: str,
        top_k: Optional[int] = None,
        min_score: float = 0.03,
    ) -> List[Dict[str, Any]]:
        return self.case_retriever.search(
            query=text,
            top_k=top_k or self.related_cases_top_k,
            min_score=min_score,
        )

    def save_feedback_case(
        self,
        case_text: str,
        predicted_outcome: str,
        requester_role: str,
        actual_outcome: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Append prediction feedback to JSONL for future retraining datasets.
        """
        payload = {
            "case_text": str(case_text or "")[:10000],
            "predicted_outcome": str(predicted_outcome or ""),
            "actual_outcome": str(actual_outcome or ""),
            "requester_role": self._normalize_requester_role(requester_role),
            "notes": str(notes or "")[:1500],
        }
        self.feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return {
            "ok": True,
            "feedback_file": str(self.feedback_store_path),
        }

    def _initialize_statute_kb(
        self,
        statute_pdf_paths: Optional[List[str | Path]],
        statute_urls: Optional[List[str]],
        enable: bool,
    ) -> Optional[StatuteKnowledgeBase]:
        """Initialize statute knowledge base from local and online sources."""
        if not enable:
            logger.info("Statute support disabled.")
            return None

        resolved_local_paths: List[str | Path] = []
        if statute_pdf_paths:
            resolved_local_paths.extend(statute_pdf_paths)

        env_pdf_path = os.getenv("LAND_ACT_PDF_PATH", "").strip()
        if env_pdf_path:
            resolved_local_paths.append(env_pdf_path)

        evidence_act_path = os.getenv("EVIDENCE_ACT_PDF_PATH", "").strip()
        if evidence_act_path:
            resolved_local_paths.append(evidence_act_path)

        # Default local paths in project data/legal/
        if DEFAULT_STATUTE_LOCAL_PATH.exists():
            resolved_local_paths.append(DEFAULT_STATUTE_LOCAL_PATH)
        if DEFAULT_LAND_USE_ACT_PATH.exists():
            resolved_local_paths.append(DEFAULT_LAND_USE_ACT_PATH)
        if DEFAULT_EVIDENCE_ACT_PATH.exists():
            resolved_local_paths.append(DEFAULT_EVIDENCE_ACT_PATH)

        # Optional comma-separated env list
        env_pdf_list = os.getenv("LAND_ACT_PDF_PATHS", "").strip()
        if env_pdf_list:
            resolved_local_paths.extend([p.strip() for p in env_pdf_list.split(",") if p.strip()])

        resolved_urls: List[str] = []
        if statute_urls:
            resolved_urls.extend([u.strip() for u in statute_urls if u and u.strip()])
        env_url_list = os.getenv("LAND_ACT_URLS", "").strip()
        if env_url_list:
            resolved_urls.extend([u.strip() for u in env_url_list.split(",") if u.strip()])
        evidence_url_list = os.getenv("EVIDENCE_ACT_URLS", "").strip()
        if evidence_url_list:
            resolved_urls.extend([u.strip() for u in evidence_url_list.split(",") if u.strip()])

        # De-duplicate sources while preserving order.
        seen_paths = set()
        deduped_local_paths: List[str | Path] = []
        for item in resolved_local_paths:
            key = str(item)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            deduped_local_paths.append(item)

        seen_urls = set()
        deduped_urls: List[str] = []
        for item in resolved_urls:
            if item in seen_urls:
                continue
            seen_urls.add(item)
            deduped_urls.append(item)

        kb = StatuteKnowledgeBase(
            local_pdf_paths=deduped_local_paths,
            online_urls=deduped_urls,
        )
        kb.load()
        status = kb.status()
        if not status["ready"]:
            logger.warning("Statute support enabled but no statute sections loaded.")
        return kb

    def statute_status(self) -> Dict[str, Any]:
        """Return statute support status for APIs/UI."""
        if self.statute_kb is None:
            return {
                "enabled": False,
                "ready": False,
                "sections_count": 0,
                "sources": [],
                "last_error": None,
            }
        status = self.statute_kb.status()
        return {
            "enabled": self.enable_statute_support,
            **status,
        }

    def list_statute_sections(self) -> List[Dict[str, Any]]:
        """List all parsed statute sections."""
        if self.statute_kb is None:
            return []
        return self.statute_kb.list_all_sections()

    def get_related_statute_sections(
        self,
        text: str,
        top_k: Optional[int] = None,
        min_score: float = 0.02,
        max_all: int = 80,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve statute sections related to the provided text."""
        if self.statute_kb is None:
            return {"top": [], "all": []}
        return self.statute_kb.get_related_sections(
            query=text,
            top_k=top_k or self.statute_top_k,
            min_score=min_score,
            max_all=max_all,
        )

    def _resolve_artifacts_path(
        self,
        explicit_artifacts_path: Optional[str | Path],
        preprocessed_data_path: Optional[str | Path],
    ) -> Path:
        """Resolve the most likely existing artifacts file path."""
        candidates = []
        if explicit_artifacts_path:
            candidates.append(Path(explicit_artifacts_path))
        if preprocessed_data_path:
            candidates.append(Path(preprocessed_data_path))

        model_dir = self.model_path.parent
        project_root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                model_dir / "inference_artifacts.pkl",
                project_root / "data" / "models" / "inference_artifacts.pkl",
                project_root / "data" / "processed" / "preprocessed_data.pkl",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        candidate_text = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(
            "Could not find inference artifacts file. Checked:\n"
            f"{candidate_text}\n"
            "Run training again to generate data/models/inference_artifacts.pkl."
        )

    def _build_feature_frame(self, cleaned_text: str):
        """Create feature frame from cleaned text."""
        sections = self.judgment_parser.parse_sections(cleaned_text)
        legal_features = self.case_annotator.extract_legal_features(cleaned_text)
        categories = self.case_annotator.categorize_land_dispute(cleaned_text)
        statute_related = self.get_related_statute_sections(cleaned_text, top_k=self.statute_top_k)
        top_statute_sections = statute_related.get("top", [])

        filtered_legal_features = {
            key: value
            for key, value in legal_features.items()
            if key in self._expected_legal_columns
        }

        full_text_for_model = cleaned_text
        if top_statute_sections:
            statute_context_lines = []
            for item in top_statute_sections:
                statute_context_lines.append(
                    f"{item.get('statute_name', 'Statute')} {item.get('section_id', '')}: "
                    f"{item.get('title', '')}. {item.get('snippet', '')}"
                )
            statute_context = "\n".join(statute_context_lines)
            full_text_for_model = (
                f"{cleaned_text}\n\n"
                f"[Relevant Statutory Context: Land Use Act + Evidence Act]\n{statute_context}"
            )

        temp_df = pd.DataFrame([
            {
                "full_text": full_text_for_model,
                "facts": sections.get("facts", ""),
                "issues": sections.get("issues", ""),
                "arguments": sections.get("arguments", ""),
                "analysis": sections.get("analysis", ""),
                "decision": sections.get("decision", ""),
                **filtered_legal_features,
            }
        ])
        return (
            temp_df,
            sections,
            filtered_legal_features,
            categories,
            statute_related,
        )

    def _decision_hint_outcome(self, cleaned_text: str, sections: Dict[str, Any]) -> Optional[str]:
        """Infer likely outcome from explicit decision wording."""
        decision_text = (sections.get("decision", "") + " " + cleaned_text[-2000:]).lower()
        success_hits = sum(hint in decision_text for hint in SUCCESS_DECISION_HINTS)
        failure_hits = sum(hint in decision_text for hint in FAILURE_DECISION_HINTS)
        if success_hits > failure_hits:
            return "granted"
        if failure_hits > success_hits:
            return "dismissed"
        return None

    def _predict_frame(
        self,
        temp_df: pd.DataFrame,
        sections: Dict[str, Any],
        legal_features: Dict[str, Any],
        categories: Any,
        statute_related: Dict[str, List[Dict[str, Any]]],
        cleaned_text: str,
        requester_role: str,
    ) -> Dict[str, Any]:
        """Predict from a prepared feature frame."""
        X, _ = self.feature_engineer.engineer_features(temp_df, fit=False)
        related_cases = self.get_related_cases(cleaned_text, top_k=self.related_cases_top_k)

        ensemble_output = self._ensemble_probability_map(X)
        ml_probability_map = ensemble_output.get("probabilities", {})
        per_model_outputs = ensemble_output.get("per_model_outputs", [])

        model_prediction_label = next(iter(ml_probability_map.keys()), "other")
        model_confidence = (
            float(next(iter(ml_probability_map.values())))
            if ml_probability_map
            else None
        )

        candidate_labels = list(ml_probability_map.keys())
        if not candidate_labels:
            candidate_labels = [canonicalize_outcome_label(str(x)) for x in self.label_encoder.classes_]
            candidate_labels = [x for x in candidate_labels if x]

        ollama_result = {"probabilities": {}, "error": None, "transport": None}
        if self.enable_ollama_hybrid and self.primary_ollama_weight > 0:
            ollama_result = self._ollama_probability_map(
                cleaned_text=cleaned_text,
                candidate_labels=candidate_labels,
                statute_related=statute_related,
            )

        ollama_probability_map = ollama_result.get("probabilities", {})
        ml_weight = self.primary_ml_weight
        ollama_weight = self.primary_ollama_weight if ollama_probability_map else 0.0
        if not self.enable_ollama_hybrid:
            ollama_weight = 0.0

        total_weight = ml_weight + ollama_weight
        if total_weight <= 0:
            ml_weight = 1.0
            ollama_weight = 0.0
            total_weight = 1.0

        ml_weight /= total_weight
        ollama_weight /= total_weight

        blended_probability_map = self._merge_probability_maps(
            [
                {"weight": ml_weight, "probabilities": ml_probability_map},
                {"weight": ollama_weight, "probabilities": ollama_probability_map},
            ]
        )
        probability_map = self._sort_probability_map(blended_probability_map)
        if not probability_map:
            probability_map = self._sort_probability_map(ml_probability_map)

        prediction_label = next(iter(probability_map.keys()), model_prediction_label)
        confidence = (
            float(next(iter(probability_map.values())))
            if probability_map
            else None
        )

        # Sanity check on explicit decision wording for land matters
        rule_override_applied = False
        override_reason = None
        hinted_outcome = self._decision_hint_outcome(cleaned_text, sections)
        rule_outcome = self.case_annotator.extract_outcome(cleaned_text, sections)
        if hinted_outcome and prediction_label != hinted_outcome:
            # Override when explicit decision wording strongly conflicts with blended output.
            low_confidence = confidence is not None and confidence < 0.60
            non_land_label = prediction_label in {"other"}
            rule_granted = rule_outcome == "PLAINTIFF_SUCCESS" and hinted_outcome == "granted"
            rule_dismissed = rule_outcome == "PLAINTIFF_FAILURE" and hinted_outcome == "dismissed"
            if low_confidence or non_land_label or rule_granted or rule_dismissed:
                prediction_label = hinted_outcome
                rule_override_applied = True
                if low_confidence:
                    override_reason = "low_confidence_decision_text_override"
                elif non_land_label:
                    override_reason = "non_land_model_label"
                elif rule_granted:
                    override_reason = "decision_text_supports_granted"
                elif rule_dismissed:
                    override_reason = "decision_text_supports_dismissed"
                confidence = None

        result = {
            "predicted_outcome": str(prediction_label),
            "confidence": float(confidence) if confidence is not None else None,
            "model_predicted_outcome": str(model_prediction_label),
            "model_confidence": float(model_confidence) if model_confidence is not None else None,
            "rule_override_applied": rule_override_applied,
            "override_reason": override_reason,
            "probabilities": probability_map,
            "raw_predicted_outcome": model_prediction_label,
            "rule_based_outcome": rule_outcome,
            "categories": [str(item) for item in categories],
            "legal_features": legal_features,
            "sections": sections,
            "related_statute_sections": statute_related.get("all", []),
            "top_statute_sections_used": statute_related.get("top", []),
            "per_statute_top_sections": statute_related.get("per_statute_top", []),
            "statute_support_used": bool(statute_related.get("top")),
            "statute_status": self.statute_status(),
            "related_cases": related_cases,
            "case_retriever_status": self.case_retriever_status(),
            "requester_role": requester_role,
            "requester_guidance": REQUESTER_ROLE_GUIDANCE.get(requester_role),
            "ensemble_model_outputs": per_model_outputs,
            "ml_ensemble_probabilities": ml_probability_map,
            "ollama_probabilities": ollama_probability_map,
            "hybrid_weights_applied": {
                "ml_models_weight": ml_weight,
                "ollama_weight": ollama_weight,
            },
            "hybrid_mode": {
                "enabled": self.enable_ollama_hybrid,
                "requested_ml_weight": self.primary_ml_weight,
                "requested_ollama_weight": self.primary_ollama_weight,
                "ollama_model": self.ollama_model,
                "ollama_error": ollama_result.get("error"),
                "ollama_transport": ollama_result.get("transport"),
            },
            "hybrid_status": self.hybrid_status(),
            "ollama_status": self.ollama_status(),
        }
        return result

    def predict_from_text(self, judgment_text: str, requester_role: str = "regular_user") -> Dict[str, Any]:
        """Predict outcome from judgment text."""
        if not judgment_text or not judgment_text.strip():
            raise ValueError("Input text is empty.")

        role = self._normalize_requester_role(requester_role)
        adapted_text = self._adapt_input_text_for_role(judgment_text, role)
        cleaned_text = self.text_preprocessor.clean(adapted_text)
        temp_df, sections, legal_features, categories, statute_related = self._build_feature_frame(cleaned_text)
        result = self._predict_frame(
            temp_df,
            sections,
            legal_features,
            categories,
            statute_related,
            cleaned_text,
            role,
        )
        result["input_type"] = "text"
        result["text_preview"] = cleaned_text[:800]
        return result

    def predict_from_file(self, file_path: str | Path, requester_role: str = "regular_user") -> Dict[str, Any]:
        """Predict outcome from PDF or DOCX file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        raw_text = self.text_extractor.extract(file_path)
        is_valid, message = self.text_extractor.validate_extraction(raw_text)
        if not is_valid:
            raise ValueError(f"Text extraction validation failed: {message}")

        role = self._normalize_requester_role(requester_role)
        adapted_text = self._adapt_input_text_for_role(raw_text, role)
        cleaned_text = self.text_preprocessor.clean(adapted_text)
        temp_df, sections, legal_features, categories, statute_related = self._build_feature_frame(cleaned_text)
        result = self._predict_frame(
            temp_df,
            sections,
            legal_features,
            categories,
            statute_related,
            cleaned_text,
            role,
        )
        result["input_type"] = "file"
        result["file_name"] = str(file_path.name)
        result["text_preview"] = cleaned_text[:800]
        return result

    def explain_prediction(self, result: Dict[str, Any]) -> str:
        """Return plain-text explanation summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("LAND MATTER PREDICTION SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Predicted Outcome: {result.get('predicted_outcome', 'N/A')}")
        if result.get("requester_role"):
            lines.append(f"Requester Role: {result.get('requester_role')}")
        confidence = result.get("confidence")
        if confidence is not None:
            lines.append(f"Confidence: {confidence * 100:.2f}%")
        hybrid_weights = result.get("hybrid_weights_applied", {})
        if hybrid_weights:
            lines.append(
                "Hybrid Weights Applied: "
                f"ML={hybrid_weights.get('ml_models_weight', 0) * 100:.1f}% | "
                f"Ollama={hybrid_weights.get('ollama_weight', 0) * 100:.1f}%"
            )

        probabilities = result.get("probabilities", {})
        if probabilities:
            lines.append("")
            lines.append("Top Probabilities:")
            for label, prob in list(probabilities.items())[:5]:
                lines.append(f"- {label}: {prob * 100:.2f}%")

        categories = result.get("categories", [])
        if categories:
            lines.append("")
            lines.append("Detected Categories:")
            for category in categories:
                lines.append(f"- {str(category).replace('_', ' ').title()}")

        legal_features = result.get("legal_features", {})
        positive_features = [name for name, value in legal_features.items() if bool(value)]
        if positive_features:
            lines.append("")
            lines.append("Detected Legal Signals:")
            for feature_name in positive_features[:12]:
                lines.append(f"- {feature_name}")

        related_statute = result.get("related_statute_sections", [])
        if related_statute:
            lines.append("")
            lines.append("Related Statutory Sections (Land Use Act + Evidence Act):")
            for section in related_statute[:10]:
                lines.append(
                    f"- {section.get('statute_name', 'Statute')} {section.get('section_id', '')}: "
                    f"{section.get('title', '')} "
                    f"(score={section.get('score', 0):.3f})"
                )

        related_cases = result.get("related_cases", [])
        if related_cases:
            lines.append("")
            lines.append("Related Trained Cases:")
            for case in related_cases[:5]:
                lines.append(
                    f"- {case.get('case_title', 'Unnamed Case')} "
                    f"(outcome={case.get('outcome', 'N/A')}, similarity={case.get('similarity', 0):.3f})"
                )

        lines.append("")
        lines.append("Disclaimer: This is a statistical prediction, not legal advice.")
        lines.append("=" * 70)
        return "\n".join(lines)

    def explain_with_ollama(
        self,
        case_text: str,
        prediction: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate plain-language explanation from local Ollama with offline-safe fallback.
        """
        top_prob_lines = []
        for label, prob in list(prediction.get("probabilities", {}).items())[:6]:
            top_prob_lines.append(f"- {label}: {prob * 100:.2f}%")

        statute_lines = []
        for section in prediction.get("top_statute_sections_used", [])[:8]:
            statute_lines.append(
                f"- {section.get('statute_name', 'Statute')} {section.get('section_id', '')}: "
                f"{section.get('title', '')} (score={section.get('score', 0):.3f})"
            )

        prompt = (
            "You are explaining an AI prediction for a Nigerian land-matter dispute.\n"
            "Do not provide legal advice.\n"
            "Keep it concise and practical.\n\n"
            f"Predicted outcome: {prediction.get('predicted_outcome')}\n"
            f"Confidence: {prediction.get('confidence')}\n"
            "Top probabilities:\n"
            f"{chr(10).join(top_prob_lines) if top_prob_lines else '- unavailable'}\n"
            "Related statutes:\n"
            f"{chr(10).join(statute_lines) if statute_lines else '- unavailable'}\n\n"
            "Case excerpt:\n"
            f"{case_text[:2000]}\n\n"
            "Respond with 4 short parts:\n"
            "1) Outcome interpretation\n"
            "2) Why model likely chose this outcome\n"
            "3) Risks/uncertainty\n"
            "4) Which statute sections should be reviewed first"
        )

        try:
            explanation = self.ollama_client.generate(prompt=prompt, model=model or self.ollama_model)
            return {
                "ok": True,
                "explanation": explanation,
                "error": None,
                "ollama_model": model or self.ollama_model,
                "transport": self.ollama_client.last_transport,
            }
        except Exception as exc:
            return {
                "ok": False,
                "explanation": None,
                "error": str(exc),
                "ollama_model": model or self.ollama_model,
                "transport": self.ollama_client.last_transport,
            }


def main() -> None:
    """CLI entrypoint for predictions."""
    parser = argparse.ArgumentParser(description="Predict land matter case outcomes")
    parser.add_argument("--file", type=str, help="Path to judgment file (PDF or DOCX)")
    parser.add_argument("--text", type=str, help="Raw judgment text")
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/best_model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="data/models/inference_artifacts.pkl",
        help="Path to inference artifacts (.pkl)",
    )
    parser.add_argument(
        "--preprocessed_data",
        type=str,
        default=None,
        help="Backward-compatible path to preprocessed_data.pkl",
    )
    parser.add_argument(
        "--training_results",
        type=str,
        default="data/models/training_results.json",
        help="Path to training_results.json for top-3 model selection",
    )
    parser.add_argument(
        "--statute_pdf",
        action="append",
        default=[],
        help="Local statute PDF path (can be provided multiple times)",
    )
    parser.add_argument(
        "--statute_url",
        action="append",
        default=[],
        help="Online statute URL (can be provided multiple times)",
    )
    parser.add_argument(
        "--statute_top_k",
        type=int,
        default=8,
        help="Top statute sections to use in prediction context",
    )
    parser.add_argument(
        "--disable_statute_support",
        action="store_true",
        help="Disable statute retrieval augmentation",
    )
    parser.add_argument(
        "--top_models_count",
        type=int,
        default=3,
        help="Number of best ML models to blend",
    )
    parser.add_argument(
        "--ml_weight",
        type=float,
        default=0.8,
        help="Global ML ensemble weight in hybrid blend",
    )
    parser.add_argument(
        "--ollama_weight",
        type=float,
        default=0.2,
        help="Global Ollama weight in hybrid blend",
    )
    parser.add_argument(
        "--disable_ollama_hybrid",
        action="store_true",
        help="Disable Ollama contribution to prediction probabilities",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model for hybrid prediction and explanations",
    )
    parser.add_argument(
        "--ollama_timeout",
        type=int,
        default=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        help="Timeout in seconds for Ollama API/CLI calls",
    )
    parser.add_argument(
        "--requester_role",
        type=str,
        default="regular_user",
        choices=["regular_user", "lawyer", "judge"],
        help="Who is submitting the case text (affects input adaptation and guidance)",
    )
    parser.add_argument(
        "--list_statute_sections",
        action="store_true",
        help="List all loaded statute sections and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON instead of plain-text summary",
    )
    args = parser.parse_args()

    if not args.file and not args.text and not args.list_statute_sections:
        print("Error: Provide --file or --text.")
        print("Example: python src/predict.py --file path/to/judgment.pdf")
        sys.exit(1)

    predictor = LandMatterPredictor(
        model_path=args.model,
        artifacts_path=args.artifacts,
        preprocessed_data_path=args.preprocessed_data,
        training_results_path=args.training_results,
        statute_pdf_paths=args.statute_pdf,
        statute_urls=args.statute_url,
        enable_statute_support=not args.disable_statute_support,
        statute_top_k=args.statute_top_k,
        top_models_count=args.top_models_count,
        ml_weight=args.ml_weight,
        ollama_weight=args.ollama_weight,
        enable_ollama_hybrid=not args.disable_ollama_hybrid,
        ollama_model=args.ollama_model,
        ollama_timeout=args.ollama_timeout,
    )

    if args.list_statute_sections:
        payload = {
            "statute_status": predictor.statute_status(),
            "sections": predictor.list_statute_sections(),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.file:
        result = predictor.predict_from_file(args.file, requester_role=args.requester_role)
    else:
        result = predictor.predict_from_text(args.text, requester_role=args.requester_role)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + predictor.explain_prediction(result) + "\n")


if __name__ == "__main__":
    main()
