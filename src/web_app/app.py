"""
Flask frontend and backend for land-matter prediction with optional Ollama reasoning.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from predict import LandMatterPredictor


MODELS_DIR = Path(os.getenv("LAND_MODELS_DIR", PROJECT_ROOT / "data" / "models"))
MODEL_PATH = Path(os.getenv("LAND_MODEL_PATH", MODELS_DIR / "best_model.pkl"))
ARTIFACTS_PATH = Path(os.getenv("LAND_ARTIFACTS_PATH", MODELS_DIR / "inference_artifacts.pkl"))
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CV_RESULTS_PATH = MODELS_DIR / "cv_results.json"
TOP_MODELS_COUNT = int(os.getenv("TOP_MODELS_COUNT", "3"))
HYBRID_ML_WEIGHT = float(os.getenv("HYBRID_ML_WEIGHT", "0.8"))
HYBRID_OLLAMA_WEIGHT = float(os.getenv("HYBRID_OLLAMA_WEIGHT", "0.2"))
DISABLE_OLLAMA_HYBRID = os.getenv("DISABLE_OLLAMA_HYBRID", "0") in {"1", "true", "yes", "on"}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "150"))
DEFAULT_LAND_ACT_PDF = PROJECT_ROOT / "data" / "legal" / "Nigerian-land-use-act-2004.pdf"
DEFAULT_EVIDENCE_ACT_PDF = PROJECT_ROOT / "data" / "legal" / "evidence-act-2011.pdf"
LAND_ACT_PDF_PATH = os.getenv("LAND_ACT_PDF_PATH", str(DEFAULT_LAND_ACT_PDF))
EVIDENCE_ACT_PDF_PATH = os.getenv("EVIDENCE_ACT_PDF_PATH", str(DEFAULT_EVIDENCE_ACT_PDF))
LAND_ACT_PDF_PATHS = [
    item.strip() for item in os.getenv("LAND_ACT_PDF_PATHS", "").split(",") if item.strip()
]
EVIDENCE_ACT_PDF_PATHS = [
    item.strip() for item in os.getenv("EVIDENCE_ACT_PDF_PATHS", "").split(",") if item.strip()
]
LAND_ACT_URLS = [
    item.strip() for item in os.getenv("LAND_ACT_URLS", "").split(",") if item.strip()
]
EVIDENCE_ACT_URLS = [
    item.strip() for item in os.getenv("EVIDENCE_ACT_URLS", "").split(",") if item.strip()
]
LAND_STATUTE_TOP_K = int(os.getenv("LAND_STATUTE_TOP_K", "8"))

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

_predictor_instance: Optional[LandMatterPredictor] = None
_predictor_error: Optional[str] = None
_retrain_lock = threading.Lock()
_retrain_state: Dict[str, Any] = {
    "running": False,
    "status": "idle",
    "message": "No retraining started yet.",
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "command": [],
    "data_files": [],
    "feedback_file": "",
    "output_tail": "",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_predictor_cache() -> None:
    global _predictor_instance, _predictor_error
    _predictor_instance = None
    _predictor_error = None


def _get_retrain_status_snapshot() -> Dict[str, Any]:
    with _retrain_lock:
        return dict(_retrain_state)


def _resolve_feedback_file() -> Path:
    env_value = os.getenv("RETRAIN_FEEDBACK_FILE", "").strip()
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate
    return MODELS_DIR / "prediction_feedback.jsonl"


def _resolve_retrain_data_files() -> List[Path]:
    env_value = os.getenv("RETRAIN_DATA_FILES", "").strip()
    if env_value:
        paths = []
        for item in [item.strip() for item in env_value.split(",") if item.strip()]:
            path = Path(item)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths.append(path)
        return [path for path in paths if path.exists()]

    preferred = [
        PROJECT_ROOT / "data" / "structured" / "cases.csv",
        PROJECT_ROOT / "data" / "structured" / "scn-cases.csv",
    ]
    existing_preferred = [path for path in preferred if path.exists()]
    if existing_preferred:
        return existing_preferred

    structured_dir = PROJECT_ROOT / "data" / "structured"
    if not structured_dir.exists():
        return []
    discovered = sorted(
        [
            path
            for path in structured_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}
        ]
    )
    return discovered


def _run_retrain_feedback_job(command: List[str], data_files: List[Path], feedback_file: Path) -> None:
    timeout_seconds = int(os.getenv("RETRAIN_TIMEOUT_SEC", "7200"))
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output_text = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        output_tail = output_text[-9000:] if output_text else ""
        with _retrain_lock:
            _retrain_state["running"] = False
            _retrain_state["finished_at"] = _utc_now_iso()
            _retrain_state["exit_code"] = int(completed.returncode)
            _retrain_state["output_tail"] = output_tail
            if completed.returncode == 0:
                _retrain_state["status"] = "completed"
                _retrain_state["message"] = "Retraining completed successfully. Models and artifacts refreshed."
            else:
                _retrain_state["status"] = "failed"
                _retrain_state["message"] = "Retraining failed. Check output_tail for details."

        if completed.returncode == 0:
            _reset_predictor_cache()
    except subprocess.TimeoutExpired:
        with _retrain_lock:
            _retrain_state["running"] = False
            _retrain_state["finished_at"] = _utc_now_iso()
            _retrain_state["status"] = "failed"
            _retrain_state["message"] = f"Retraining timed out after {timeout_seconds} seconds."
            _retrain_state["exit_code"] = None
    except Exception as exc:
        with _retrain_lock:
            _retrain_state["running"] = False
            _retrain_state["finished_at"] = _utc_now_iso()
            _retrain_state["status"] = "failed"
            _retrain_state["message"] = f"Retraining crashed: {exc}"
            _retrain_state["exit_code"] = None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_artifacts() -> Dict[str, Any]:
    training_results = _load_json(TRAINING_RESULTS_PATH)
    metadata = _load_json(MODEL_METADATA_PATH)
    cv_results = _load_json(CV_RESULTS_PATH)
    return {
        "training_results": training_results,
        "metadata": metadata,
        "cv_results": cv_results,
        "paths": {
            "model_path": str(MODEL_PATH),
            "artifacts_path": str(ARTIFACTS_PATH),
            "training_results": str(TRAINING_RESULTS_PATH),
            "metadata": str(MODEL_METADATA_PATH),
            "cv_results": str(CV_RESULTS_PATH),
        },
    }


def get_predictor() -> LandMatterPredictor:
    global _predictor_instance, _predictor_error
    if _predictor_instance is not None:
        return _predictor_instance

    if _predictor_error:
        raise RuntimeError(_predictor_error)

    try:
        statute_pdf_paths = []
        if LAND_ACT_PDF_PATH:
            statute_pdf_paths.append(LAND_ACT_PDF_PATH)
        if EVIDENCE_ACT_PDF_PATH:
            statute_pdf_paths.append(EVIDENCE_ACT_PDF_PATH)
        statute_pdf_paths.extend(LAND_ACT_PDF_PATHS)
        statute_pdf_paths.extend(EVIDENCE_ACT_PDF_PATHS)
        statute_urls = []
        statute_urls.extend(LAND_ACT_URLS)
        statute_urls.extend(EVIDENCE_ACT_URLS)

        _predictor_instance = LandMatterPredictor(
            model_path=MODEL_PATH,
            artifacts_path=ARTIFACTS_PATH,
            training_results_path=TRAINING_RESULTS_PATH,
            statute_pdf_paths=statute_pdf_paths,
            statute_urls=statute_urls,
            statute_top_k=LAND_STATUTE_TOP_K,
            top_models_count=TOP_MODELS_COUNT,
            ml_weight=HYBRID_ML_WEIGHT,
            ollama_weight=HYBRID_OLLAMA_WEIGHT,
            enable_ollama_hybrid=not DISABLE_OLLAMA_HYBRID,
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
        )
        return _predictor_instance
    except Exception as exc:
        _predictor_error = str(exc)
        raise RuntimeError(_predictor_error) from exc


def predictor_status() -> Dict[str, Any]:
    try:
        predictor = get_predictor()
        classes = list(getattr(predictor.label_encoder, "classes_", []))
        statute = predictor.statute_status()
        hybrid = predictor.hybrid_status()
        ollama = predictor.ollama_status()
        roles = predictor.requester_role_help()
        case_retriever = predictor.case_retriever_status()
        return {
            "ready": True,
            "error": None,
            "classes": [str(label) for label in classes],
            "model_path": str(MODEL_PATH),
            "artifacts_path": str(ARTIFACTS_PATH),
            "statute": statute,
            "hybrid": hybrid,
            "ollama": ollama,
            "requester_roles": roles,
            "case_retriever": case_retriever,
            "feedback_store_path": str(predictor.feedback_store_path),
        }
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
            "classes": [],
            "model_path": str(MODEL_PATH),
            "artifacts_path": str(ARTIFACTS_PATH),
            "statute": {
                "enabled": True,
                "ready": False,
                "sections_count": 0,
                "sources": [],
                "last_error": str(exc),
            },
            "hybrid": {
                "top_models_count": 0,
                "ml_weight": HYBRID_ML_WEIGHT,
                "ollama_weight": HYBRID_OLLAMA_WEIGHT,
                "ollama_hybrid_enabled": not DISABLE_OLLAMA_HYBRID,
                "ensemble_models": [],
            },
            "requester_roles": {
                "regular_user": "Plain language accepted.",
                "lawyer": "Use legal framing with issues and reliefs.",
                "judge": "Use judicial issue-analysis-order format.",
            },
            "case_retriever": {
                "ready": False,
                "records_count": 0,
            },
            "feedback_store_path": str(MODELS_DIR / "prediction_feedback.jsonl"),
            "ollama": {
                "available": False,
                "base_url": OLLAMA_BASE_URL,
                "default_model": OLLAMA_MODEL,
                "models": [],
                "cli_available": False,
                "error": str(exc),
            },
        }


def maybe_add_ollama_explanation(
    response_payload: Dict[str, Any],
    predictor: LandMatterPredictor,
    prediction: Dict[str, Any],
    source_text: str,
    use_ollama: bool,
    requested_model: Optional[str],
) -> None:
    if not use_ollama:
        response_payload["ollama_used"] = False
        response_payload["ollama_explanation"] = None
        response_payload["ollama_error"] = None
        return

    explanation_payload = predictor.explain_with_ollama(
        case_text=source_text,
        prediction=prediction,
        model=requested_model,
    )
    if explanation_payload.get("ok"):
        response_payload["ollama_used"] = True
        response_payload["ollama_model"] = explanation_payload.get("ollama_model")
        response_payload["ollama_transport"] = explanation_payload.get("transport")
        response_payload["ollama_explanation"] = explanation_payload.get("explanation")
        response_payload["ollama_error"] = None
    else:
        response_payload["ollama_used"] = False
        response_payload["ollama_model"] = explanation_payload.get("ollama_model")
        response_payload["ollama_transport"] = explanation_payload.get("transport")
        response_payload["ollama_explanation"] = None
        response_payload["ollama_error"] = explanation_payload.get("error")


@app.get("/health")
def health():
    artifacts = load_artifacts()
    status = predictor_status()
    ollama_state = status.get("ollama", {})
    ollama_available = bool(ollama_state.get("available", False))
    statute_status = status.get("statute", {})
    hybrid_status = status.get("hybrid", {})
    case_retriever_status = status.get("case_retriever", {})
    retrain_state = _get_retrain_status_snapshot()
    return jsonify(
        {
            "status": "ok",
            "predictor_ready": status["ready"],
            "predictor_error": status["error"],
            "ollama_available": ollama_available,
            "ollama_cli_available": bool(ollama_state.get("cli_available", False)),
            "statute_enabled": statute_status.get("enabled", False),
            "statute_ready": statute_status.get("ready", False),
            "statute_sections_count": statute_status.get("sections_count", 0),
            "hybrid_enabled": hybrid_status.get("ollama_hybrid_enabled", False),
            "hybrid_top_models_count": hybrid_status.get("top_models_count", 0),
            "case_retriever_ready": case_retriever_status.get("ready", False),
            "case_retriever_records_count": case_retriever_status.get("records_count", 0),
            "retrain_running": retrain_state.get("running", False),
            "retrain_status": retrain_state.get("status", "idle"),
            "training_results_found": bool(artifacts["training_results"]),
            "metadata_found": bool(artifacts["metadata"]),
            "cv_results_found": bool(artifacts["cv_results"]),
        }
    )


@app.get("/api/config")
def api_config():
    status = predictor_status()
    return jsonify(
        {
            "predictor": status,
            "statute": status.get("statute", {}),
            "hybrid": status.get("hybrid", {}),
            "requester_roles": status.get("requester_roles", {}),
            "case_retriever": status.get("case_retriever", {}),
            "statute_config": {
                "default_pdf_path": LAND_ACT_PDF_PATH,
                "extra_pdf_paths": LAND_ACT_PDF_PATHS,
                "default_evidence_pdf_path": EVIDENCE_ACT_PDF_PATH,
                "extra_evidence_pdf_paths": EVIDENCE_ACT_PDF_PATHS,
                "urls": LAND_ACT_URLS + EVIDENCE_ACT_URLS,
                "top_k": LAND_STATUTE_TOP_K,
            },
            "ollama": status.get("ollama", {}),
            "retrain": _get_retrain_status_snapshot(),
            "paths": load_artifacts()["paths"],
        }
    )


@app.get("/api/results")
def api_results():
    return jsonify(load_artifacts())


@app.get("/api/requester/roles")
def api_requester_roles():
    try:
        predictor = get_predictor()
        return jsonify(
            {
                "ok": True,
                "roles": predictor.requester_role_help(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/ollama/models")
def api_ollama_models():
    try:
        predictor = get_predictor()
        ollama = predictor.ollama_status()
        return jsonify(ollama)
    except Exception as exc:
        return jsonify({"available": False, "models": [], "error": str(exc), "cli_available": False}), 503


@app.get("/api/statute/sections")
def api_statute_sections():
    """
    List all statute sections or return sections related to query `q`.
    Query params:
      - q: query text (optional)
      - top_k: integer for related search (optional)
      - min_score: float threshold (optional)
      - limit: max results for full list (optional, default 300)
    """
    try:
        predictor = get_predictor()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    q = str(request.args.get("q", "")).strip()
    top_k = int(request.args.get("top_k", LAND_STATUTE_TOP_K))
    min_score = float(request.args.get("min_score", 0.02))
    limit = int(request.args.get("limit", 1000))

    if q:
        related = predictor.get_related_statute_sections(
            text=q,
            top_k=top_k,
            min_score=min_score,
            max_all=limit,
        )
        return jsonify(
            {
                "ok": True,
                "query": q,
                "statute_status": predictor.statute_status(),
                "top_sections": related.get("top", []),
                "related_sections": related.get("all", []),
            }
        )

    sections = predictor.list_statute_sections()[:limit]
    return jsonify(
        {
            "ok": True,
            "query": None,
            "statute_status": predictor.statute_status(),
            "sections": sections,
        }
    )


@app.get("/api/cases/related")
def api_related_cases():
    try:
        predictor = get_predictor()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    q = str(request.args.get("q", "")).strip()
    top_k = int(request.args.get("top_k", 5))
    min_score = float(request.args.get("min_score", 0.03))
    if not q:
        return jsonify({"ok": False, "error": "Provide query parameter q"}), 400

    related_cases = predictor.get_related_cases(text=q, top_k=top_k, min_score=min_score)
    return jsonify(
        {
            "ok": True,
            "query": q,
            "case_retriever_status": predictor.case_retriever_status(),
            "related_cases": related_cases,
        }
    )


@app.post("/api/predict/text")
def api_predict_text():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    use_ollama = bool(payload.get("use_ollama", False))
    ollama_model = str(payload.get("ollama_model", "")).strip() or None
    requester_role = str(payload.get("requester_role", "regular_user")).strip() or "regular_user"

    if not text:
        return jsonify({"ok": False, "error": "Provide non-empty text input."}), 400

    try:
        predictor = get_predictor()
        prediction = predictor.predict_from_text(text, requester_role=requester_role)
        response_payload: Dict[str, Any] = {"ok": True, "prediction": prediction}
        maybe_add_ollama_explanation(
            response_payload=response_payload,
            predictor=predictor,
            prediction=prediction,
            source_text=text,
            use_ollama=use_ollama,
            requested_model=ollama_model,
        )
        return jsonify(response_payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/predict/file")
def api_predict_file():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename is None or uploaded_file.filename.strip() == "":
        return jsonify({"ok": False, "error": "Attach a PDF or DOCX file."}), 400

    suffix = Path(uploaded_file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify({"ok": False, "error": "Only .pdf and .docx files are supported."}), 400

    use_ollama = str(request.form.get("use_ollama", "false")).lower() in {"1", "true", "yes", "on"}
    ollama_model = str(request.form.get("ollama_model", "")).strip() or None
    requester_role = str(request.form.get("requester_role", "regular_user")).strip() or "regular_user"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = Path(temp_file.name)

        predictor = get_predictor()
        prediction = predictor.predict_from_file(temp_path, requester_role=requester_role)
        response_payload: Dict[str, Any] = {"ok": True, "prediction": prediction}
        source_text = str(prediction.get("text_preview", ""))
        maybe_add_ollama_explanation(
            response_payload=response_payload,
            predictor=predictor,
            prediction=prediction,
            source_text=source_text,
            use_ollama=use_ollama,
            requested_model=ollama_model,
        )
        return jsonify(response_payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@app.post("/api/feedback")
def api_feedback():
    payload = request.get_json(silent=True) or {}
    case_text = str(payload.get("case_text", "")).strip()
    predicted_outcome = str(payload.get("predicted_outcome", "")).strip()
    actual_outcome = str(payload.get("actual_outcome", "")).strip()
    requester_role = str(payload.get("requester_role", "regular_user")).strip()
    notes = str(payload.get("notes", "")).strip()

    if not case_text:
        return jsonify({"ok": False, "error": "case_text is required"}), 400
    if not predicted_outcome:
        return jsonify({"ok": False, "error": "predicted_outcome is required"}), 400

    try:
        predictor = get_predictor()
        saved = predictor.save_feedback_case(
            case_text=case_text,
            predicted_outcome=predicted_outcome,
            requester_role=requester_role,
            actual_outcome=actual_outcome,
            notes=notes,
        )
        return jsonify({"ok": True, **saved})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/retrain/status")
def api_retrain_status():
    return jsonify({"ok": True, "retrain": _get_retrain_status_snapshot()})


@app.post("/api/retrain/feedback")
def api_retrain_from_feedback():
    data_files = _resolve_retrain_data_files()
    if not data_files:
        return jsonify(
            {
                "ok": False,
                "error": "No structured training files found. Set RETRAIN_DATA_FILES or add files in data/structured.",
            }
        ), 400

    feedback_file = _resolve_feedback_file()
    if not feedback_file.exists():
        return jsonify(
            {
                "ok": False,
                "error": f"Feedback file not found: {feedback_file}",
            }
        ), 400

    with _retrain_lock:
        if _retrain_state.get("running"):
            return jsonify({"ok": False, "error": "Retraining is already running.", "retrain": dict(_retrain_state)}), 409

        command = [
            sys.executable,
            str(SRC_ROOT / "main_training.py"),
            "--data_file",
            *[str(path) for path in data_files],
            "--output_dir",
            str(MODELS_DIR),
            "--include_feedback",
            "--feedback_file",
            str(feedback_file),
            "--n_jobs",
            "1",
        ]

        _retrain_state["running"] = True
        _retrain_state["status"] = "running"
        _retrain_state["message"] = "Retraining started."
        _retrain_state["started_at"] = _utc_now_iso()
        _retrain_state["finished_at"] = None
        _retrain_state["exit_code"] = None
        _retrain_state["command"] = command
        _retrain_state["data_files"] = [str(path) for path in data_files]
        _retrain_state["feedback_file"] = str(feedback_file)
        _retrain_state["output_tail"] = ""

    worker = threading.Thread(
        target=_run_retrain_feedback_job,
        args=(command, data_files, feedback_file),
        daemon=True,
    )
    worker.start()

    return jsonify({"ok": True, "retrain": _get_retrain_status_snapshot()})


@app.get("/")
def dashboard():
    artifacts = load_artifacts()
    training_results = artifacts["training_results"]
    metadata = artifacts["metadata"]
    cv_results = artifacts["cv_results"]
    predict_status = predictor_status()
    statute_status = predict_status.get("statute", {})
    case_retriever_status = predict_status.get("case_retriever", {})
    retrain_state = _get_retrain_status_snapshot()

    training_rows = []
    for model_name, metrics in training_results.items():
        training_rows.append(
            {
                "model": model_name,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
            }
        )
    training_rows.sort(key=lambda row: row["accuracy"], reverse=True)

    ollama_state = predict_status.get("ollama", {})
    ollama_models = ollama_state.get("models", []) if isinstance(ollama_state, dict) else []
    ollama_error = ollama_state.get("error") if isinstance(ollama_state, dict) else None

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ADEMOLA DANIEL Land-Matter Predictor</title>
  <style>
    :root {
      --paper: #f8f5ee;
      --ink: #1d2a35;
      --card: #ffffff;
      --edge: #d9d0c2;
      --accent: #0f766e;
      --accent-soft: #d7f0ec;
      --warning: #9a3412;
      --warning-soft: #ffedd5;
      --ok: #166534;
      --ok-soft: #dcfce7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 20% 10%, #e6efe9 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #f0ece1 0%, transparent 32%),
        linear-gradient(165deg, #f7f3ea 0%, #f3f7f8 100%);
      min-height: 100vh;
    }
    .page {
      max-width: 1180px;
      margin: 24px auto 40px auto;
      padding: 0 16px;
    }
    .hero {
      background: var(--card);
      border: 1px solid var(--edge);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 28px rgba(12, 34, 44, 0.08);
      margin-bottom: 16px;
    }
    .title {
      margin: 0;
      font-size: clamp(1.3rem, 2.8vw, 2rem);
      letter-spacing: 0.02em;
    }
    .subtitle {
      margin-top: 8px;
      color: #3f4c56;
      font-size: 0.96rem;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 16px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--edge);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(12, 34, 44, 0.06);
    }
    .panel h2 {
      margin: 0 0 12px 0;
      font-size: 1.08rem;
    }
    .status {
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 0.92rem;
      margin-bottom: 10px;
    }
    .status.ok {
      background: var(--ok-soft);
      color: var(--ok);
      border: 1px solid #86efac;
    }
    .status.warn {
      background: var(--warning-soft);
      color: var(--warning);
      border: 1px solid #fdba74;
    }
    .field { margin-bottom: 12px; }
    label {
      display: block;
      font-size: 0.9rem;
      margin-bottom: 6px;
      color: #334155;
      font-weight: 600;
    }
    textarea, input[type="text"], input[type="file"], select {
      width: 100%;
      border: 1px solid #c7cfd9;
      background: #ffffff;
      border-radius: 10px;
      padding: 10px;
      font-size: 0.95rem;
      color: var(--ink);
    }
    textarea {
      min-height: 170px;
      resize: vertical;
      line-height: 1.4;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
    }
    .tab-btn {
      border: 1px solid #98a5b6;
      background: #ffffff;
      border-radius: 999px;
      padding: 7px 13px;
      cursor: pointer;
      font-size: 0.9rem;
    }
    .tab-btn.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #0f4d47;
      font-weight: 700;
    }
    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 6px;
      flex-wrap: wrap;
    }
    .btn {
      border: 1px solid #0f766e;
      background: linear-gradient(180deg, #14b8a6, #0f766e);
      color: #ffffff;
      border-radius: 12px;
      padding: 9px 16px;
      cursor: pointer;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .tiny { font-size: 0.84rem; color: #4b5563; }
    .result-pill {
      display: inline-block;
      background: #ecfeff;
      border: 1px solid #7dd3fc;
      color: #155e75;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.84rem;
      margin-right: 8px;
      margin-top: 6px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 0.92rem;
    }
    th, td {
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      padding: 8px 6px;
    }
    th { color: #475569; font-weight: 700; }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 12px;
      font-family: "Consolas", "Courier New", monospace;
      font-size: 0.86rem;
      line-height: 1.35;
    }
    .hidden { display: none; }
    .cards {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .metric {
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 10px;
      background: #fbfdff;
    }
    .metric .label { font-size: 0.8rem; color: #64748b; }
    .metric .value { font-size: 1.1rem; margin-top: 4px; font-weight: 700; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .cards { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 680px) {
      .row { grid-template-columns: 1fr; }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1 class="title">ADEMOLA DANIEL Land-Matter Prediction Workbench</h1>
      <div class="subtitle">
       I Use trained ML models to predict outcomes from case text or uploaded judgments, then optionally request a plain-language explanation from Ollama for better description.
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Prediction Input</h2>
        {% if predictor_ready %}
        <div class="status ok">
          Predictor ready. Model: <strong>{{ model_path }}</strong>
        </div>
        {% else %}
        <div class="status warn">
          Predictor not ready: {{ predictor_error }}
        </div>
        {% endif %}

        <div class="tabs">
          <button id="tabText" class="tab-btn active" type="button">Text Input</button>
          <button id="tabFile" class="tab-btn" type="button">File Upload</button>
        </div>

        <div id="textSection">
          <div class="field">
            <label for="caseText">Case Text</label>
            <textarea id="caseText" placeholder="Paste case facts, issues, arguments, and decision text here."></textarea>
          </div>
        </div>

        <div id="fileSection" class="hidden">
          <div class="field">
            <label for="caseFile">Upload PDF or DOCX</label>
            <input id="caseFile" type="file" accept=".pdf,.docx" />
          </div>
        </div>

        <div class="field">
          <label for="requesterRole">Requester Type</label>
          <select id="requesterRole">
            {% for role_key, role_help in requester_roles.items() %}
            <option value="{{ role_key }}" data-help="{{ role_help }}">{{ role_key.replace('_', ' ').title() }}</option>
            {% endfor %}
          </select>
          <div id="roleHelp" class="tiny" style="margin-top:4px;">
            {{ requester_roles.get("regular_user", "") }}
          </div>
        </div>

        <div class="row">
          <div class="field">
            <label for="useOllama">
              <input id="useOllama" type="checkbox" style="width:auto; margin-right:6px;" />
              Add Ollama Explanation
            </label>
          </div>
          <div class="field">
            <label for="ollamaModel">Ollama Model (optional)</label>
            <input id="ollamaModel" type="text" placeholder="e.g. llama3" value="{{ default_ollama_model }}" />
          </div>
        </div>

        <div class="actions">
          <button id="predictBtn" class="btn" type="button" {% if not predictor_ready %}disabled{% endif %}>
            Run Prediction
          </button>
          <span id="requestStatus" class="tiny">Ready.</span>
        </div>

        <div class="field" style="margin-top:12px;">
          <div class="tiny"><strong>Ollama Status:</strong>
            {% if ollama_models %}
              connected ({{ ollama_models|length }} model(s) found)
            {% else %}
              {% if ollama_error %}not connected: {{ ollama_error }}{% else %}not connected{% endif %}
            {% endif %}
          </div>
          <div class="tiny" style="margin-top:4px;"><strong>Statute Support:</strong>
            {% if statute_status.get("ready") %}
              ready ({{ statute_status.get("sections_count", 0) }} sections loaded)
              {% if statute_status.get("statutes_loaded") %}
                - {{ statute_status.get("statutes_loaded")|join(", ") }}
              {% endif %}
            {% else %}
              unavailable{% if statute_status.get("last_error") %}: {{ statute_status.get("last_error") }}{% endif %}
            {% endif %}
          </div>
          <div class="tiny" style="margin-top:4px;"><strong>Hybrid Mode:</strong>
            {% if hybrid_status %}
              top {{ hybrid_status.get("top_models_count", 0) }} models,
              ML {{ (hybrid_status.get("ml_weight", 0) * 100)|round(1) }}% /
              Ollama {{ (hybrid_status.get("ollama_weight", 0) * 100)|round(1) }}%
            {% else %}
              unavailable
            {% endif %}
          </div>
          <div class="tiny" style="margin-top:4px;"><strong>Case Citations:</strong>
            {% if case_retriever_status.get("ready") %}
              ready ({{ case_retriever_status.get("records_count", 0) }} trained cases indexed)
            {% else %}
              unavailable (re-train to generate `case_reference_records`)
            {% endif %}
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Prediction Output</h2>
        <div id="errorBox" class="status warn hidden"></div>

        <div id="resultArea" class="hidden">
          <div class="cards">
            <div class="metric">
              <div class="label">Predicted Outcome</div>
              <div id="predictedOutcome" class="value">-</div>
            </div>
            <div class="metric">
              <div class="label">Confidence</div>
              <div id="predictedConfidence" class="value">-</div>
            </div>
          </div>
          <div id="overrideInfo" class="tiny" style="margin-top:8px;"></div>

          <h3 style="margin:14px 0 8px 0;">Top Probabilities</h3>
          <table id="probabilityTable">
            <thead><tr><th>Outcome</th><th>Probability</th></tr></thead>
            <tbody></tbody>
          </table>

          <h3 style="margin:14px 0 8px 0;">Detected Categories</h3>
          <div id="categoriesBox"></div>

          <h3 style="margin:14px 0 8px 0;">Detected Legal Features</h3>
          <div id="legalBox"></div>

          <h3 style="margin:14px 0 8px 0;">Related Statute Sections (Land Use Act + Evidence Act)</h3>
          <div id="statuteBox" style="height:500px; overflow-y:scroll;"></div>

          <h3 style="margin:14px 0 8px 0;">Related Trained Cases</h3>
          <div id="casesBox"></div>

          <h3 style="margin:14px 0 8px 0;">Improve With Feedback (Optional)</h3>
          <div class="row">
            <div class="field">
              <label for="actualOutcome">Actual Outcome</label>
              <input id="actualOutcome" type="text" placeholder="e.g. granted, dismissed, matter remitted" />
            </div>
            <div class="field">
              <label for="feedbackNotes">Notes</label>
              <input id="feedbackNotes" type="text" placeholder="Optional correction notes" />
            </div>
          </div>
          <div class="actions">
            <button id="feedbackBtn" class="btn" type="button">Save Feedback</button>
            <span id="feedbackStatus" class="tiny">No feedback saved yet.</span>
          </div>
        </div>

        <div id="ollamaArea" class="hidden" style="margin-top: 14px;">
          <h3 style="margin:0 0 8px 0;">Ollama Explanation</h3>
          <pre id="ollamaText"></pre>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top:16px;">
      <h2>Training Summary</h2>
      {% if training_rows %}
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {% for row in training_rows %}
          <tr>
            <td>{{ row.model }}</td>
            <td>{{ "%.4f"|format(row.accuracy) }}</td>
            <td>{{ "%.4f"|format(row.precision|float) if row.precision is not none else "N/A" }}</td>
            <td>{{ "%.4f"|format(row.recall|float) if row.recall is not none else "N/A" }}</td>
            <td>{{ "%.4f"|format(row.f1_score|float) if row.f1_score is not none else "N/A" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <div class="tiny">
        No training metrics found. Run:
        <code>python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models</code>
      </div>
      {% endif %}
      {% if metadata %}
      <div class="tiny" style="margin-top:10px;">
        Best model: <strong>{{ metadata.get("best_model_name", "N/A") }}</strong>,
        accuracy: <strong>{{ metadata.get("best_accuracy", "N/A") }}</strong>,
        features: <strong>{{ metadata.get("feature_count", "N/A") }}</strong>
      </div>
      {% endif %}
      {% if cv_results %}
      <div class="tiny" style="margin-top:6px;">
        CV accuracy: {{ cv_results.get("accuracy_mean", "N/A") }} (+/- {{ cv_results.get("accuracy_std", "N/A") }})
      </div>
      {% endif %}
      <div class="actions" style="margin-top:12px;">
        <button id="retrainBtn" class="btn" type="button">Retrain From Feedback</button>
        <span id="retrainStatus" class="tiny">
          {{ retrain_state.get("message", "No retraining started yet.") }}
        </span>
      </div>
      <pre id="retrainLog" class="hidden" style="margin-top:8px;"></pre>
    </section>
  </div>

  <script>
    const tabText = document.getElementById("tabText");
    const tabFile = document.getElementById("tabFile");
    const textSection = document.getElementById("textSection");
    const fileSection = document.getElementById("fileSection");
    const predictBtn = document.getElementById("predictBtn");
    const statusNode = document.getElementById("requestStatus");
    const errorBox = document.getElementById("errorBox");
    const resultArea = document.getElementById("resultArea");
    const ollamaArea = document.getElementById("ollamaArea");
    const ollamaText = document.getElementById("ollamaText");
    const useOllama = document.getElementById("useOllama");
    const ollamaModel = document.getElementById("ollamaModel");
    const caseText = document.getElementById("caseText");
    const caseFile = document.getElementById("caseFile");
    const requesterRole = document.getElementById("requesterRole");
    const roleHelp = document.getElementById("roleHelp");
    const actualOutcome = document.getElementById("actualOutcome");
    const feedbackNotes = document.getElementById("feedbackNotes");
    const feedbackBtn = document.getElementById("feedbackBtn");
    const feedbackStatus = document.getElementById("feedbackStatus");
    const retrainBtn = document.getElementById("retrainBtn");
    const retrainStatus = document.getElementById("retrainStatus");
    const retrainLog = document.getElementById("retrainLog");

    const predictedOutcome = document.getElementById("predictedOutcome");
    const predictedConfidence = document.getElementById("predictedConfidence");
    const probabilityBody = document.querySelector("#probabilityTable tbody");
    const categoriesBox = document.getElementById("categoriesBox");
    const legalBox = document.getElementById("legalBox");
    const statuteBox = document.getElementById("statuteBox");
    const casesBox = document.getElementById("casesBox");
    const overrideInfo = document.getElementById("overrideInfo");

    let inputMode = "text";
    let latestPrediction = null;
    let latestCaseText = "";
    let retrainPollHandle = null;

    function setMode(mode) {
      inputMode = mode;
      const textActive = mode === "text";
      tabText.classList.toggle("active", textActive);
      tabFile.classList.toggle("active", !textActive);
      textSection.classList.toggle("hidden", !textActive);
      fileSection.classList.toggle("hidden", textActive);
      clearError();
    }

    function clearError() {
      errorBox.classList.add("hidden");
      errorBox.textContent = "";
    }

    function showError(message) {
      errorBox.classList.remove("hidden");
      errorBox.textContent = message;
    }

    function setLoading(loading, message) {
      predictBtn.disabled = loading;
      statusNode.textContent = message;
    }

    function updateRoleHelp() {
      const selected = requesterRole.options[requesterRole.selectedIndex];
      roleHelp.textContent = selected ? (selected.dataset.help || "") : "";
    }

    function setRetrainUI(state) {
      const running = Boolean(state && state.running);
      retrainBtn.disabled = running;
      retrainStatus.textContent = (state && state.message) ? state.message : "No retraining started yet.";
      if (state && state.output_tail) {
        retrainLog.classList.remove("hidden");
        retrainLog.textContent = state.output_tail;
      } else {
        retrainLog.classList.add("hidden");
        retrainLog.textContent = "";
      }
      if (!running && retrainPollHandle !== null) {
        clearInterval(retrainPollHandle);
        retrainPollHandle = null;
      }
    }

    async function fetchRetrainStatus() {
      try {
        const response = await fetch("/api/retrain/status");
        const payload = await response.json();
        if (response.ok && payload.ok) {
          setRetrainUI(payload.retrain || {});
        }
      } catch (_) {
        // Keep current UI state if polling fails briefly.
      }
    }

    async function startRetrainFromFeedback() {
      try {
        retrainBtn.disabled = true;
        retrainStatus.textContent = "Starting retraining...";
        const response = await fetch("/api/retrain/feedback", { method: "POST" });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Retraining could not start.");
        }
        setRetrainUI(payload.retrain || {});
        if (retrainPollHandle !== null) {
          clearInterval(retrainPollHandle);
        }
        retrainPollHandle = setInterval(fetchRetrainStatus, 5000);
      } catch (err) {
        retrainBtn.disabled = false;
        retrainStatus.textContent = "Retrain error: " + (err.message || String(err));
      }
    }

    function renderPrediction(prediction) {
      latestPrediction = prediction;
      resultArea.classList.remove("hidden");
      predictedOutcome.textContent = prediction.predicted_outcome || "-";
      if (prediction.confidence === null || prediction.confidence === undefined) {
        predictedConfidence.textContent = "N/A";
      } else {
        predictedConfidence.textContent = (prediction.confidence * 100).toFixed(2) + "%";
      }

      if (prediction.rule_override_applied) {
        const modelConfidenceText = (prediction.model_confidence === null || prediction.model_confidence === undefined)
          ? "N/A"
          : (prediction.model_confidence * 100).toFixed(2) + "%";
        const appliedWeights = prediction.hybrid_weights_applied || {};
        const appliedMl = Number(appliedWeights.ml_models_weight || 0) * 100;
        const appliedOllama = Number(appliedWeights.ollama_weight || 0) * 100;
        overrideInfo.textContent =
          "Rule override applied (" + (prediction.override_reason || "decision_sanity_check") +
          "). Model predicted '" + (prediction.model_predicted_outcome || "-") +
          "' at " + modelConfidenceText + ". Hybrid blend used ML " +
          appliedMl.toFixed(1) + "% / Ollama " + appliedOllama.toFixed(1) + "%.";
      } else {
        const appliedWeights = prediction.hybrid_weights_applied || {};
        const appliedMl = Number(appliedWeights.ml_models_weight || 0) * 100;
        const appliedOllama = Number(appliedWeights.ollama_weight || 0) * 100;
        const hybridInfo = "Hybrid blend used ML " + appliedMl.toFixed(1) + "% / Ollama " + appliedOllama.toFixed(1) + "%.";
        const ollamaError = (prediction.hybrid_mode && prediction.hybrid_mode.ollama_error)
          ? (" Ollama blend fallback reason: " + prediction.hybrid_mode.ollama_error)
          : "";
        overrideInfo.textContent = hybridInfo + ollamaError;
      }

      probabilityBody.innerHTML = "";
      const probabilities = prediction.probabilities || {};
      Object.entries(probabilities).forEach(([label, value]) => {
        const row = document.createElement("tr");
        const labelTd = document.createElement("td");
        const valueTd = document.createElement("td");
        labelTd.textContent = label;
        valueTd.textContent = (value * 100).toFixed(2) + "%";
        row.appendChild(labelTd);
        row.appendChild(valueTd);
        probabilityBody.appendChild(row);
      });

      categoriesBox.innerHTML = "";
      (prediction.categories || []).forEach((category) => {
        const chip = document.createElement("span");
        chip.className = "result-pill";
        chip.textContent = String(category).replaceAll("_", " ");
        categoriesBox.appendChild(chip);
      });

      legalBox.innerHTML = "";
      const legalFeatures = prediction.legal_features || {};
      const positives = Object.entries(legalFeatures).filter(([_, value]) => Boolean(value));
      if (!positives.length) {
        const none = document.createElement("span");
        none.className = "tiny";
        none.textContent = "No legal feature flags detected.";
        legalBox.appendChild(none);
      } else {
        positives.forEach(([name]) => {
          const chip = document.createElement("span");
          chip.className = "result-pill";
          chip.textContent = name;
          legalBox.appendChild(chip);
        });
      }

      statuteBox.innerHTML = "";
      const relatedSections = prediction.related_statute_sections || [];
      if (!relatedSections.length) {
        const none = document.createElement("span");
        none.className = "tiny";
        none.textContent = "No related statute sections detected.";
        statuteBox.appendChild(none);
      } else {
        relatedSections.forEach((section) => {
          const wrapper = document.createElement("div");
          wrapper.style.marginBottom = "10px";
          const heading = document.createElement("div");
          heading.className = "tiny";
          heading.style.fontWeight = "700";
          heading.textContent =
            (section.statute_name ? section.statute_name + " " : "") +
            (section.section_id || "Section") +
            ": " + (section.title || "") +
            " (score=" + Number(section.score || 0).toFixed(3) + ")";
          const snippet = document.createElement("pre");
          snippet.textContent = section.snippet || section.text || "";
          wrapper.appendChild(heading);
          wrapper.appendChild(snippet);
          statuteBox.appendChild(wrapper);
        });
      }

      casesBox.innerHTML = "";
      const relatedCases = prediction.related_cases || [];
      if (!relatedCases.length) {
        const none = document.createElement("span");
        none.className = "tiny";
        none.textContent = "No related trained cases available. Re-train to generate citation index.";
        casesBox.appendChild(none);
      } else {
        relatedCases.forEach((item) => {
          const wrapper = document.createElement("div");
          wrapper.style.marginBottom = "10px";
          const heading = document.createElement("div");
          heading.className = "tiny";
          heading.style.fontWeight = "700";
          heading.textContent =
            (item.case_title || "Unnamed Case") +
            " | outcome=" + (item.outcome || "N/A") +
            " | similarity=" + Number(item.similarity || 0).toFixed(3);
          const snippet = document.createElement("pre");
          snippet.textContent = item.snippet || "";
          wrapper.appendChild(heading);
          wrapper.appendChild(snippet);
          casesBox.appendChild(wrapper);
        });
      }
    }

    async function submitPrediction() {
      clearError();
      resultArea.classList.add("hidden");
      ollamaArea.classList.add("hidden");
      ollamaText.textContent = "";
      updateRoleHelp();

      const shouldUseOllama = useOllama.checked;
      const requestedModel = (ollamaModel.value || "").trim();
      const role = (requesterRole.value || "regular_user").trim();
      latestPrediction = null;
      latestCaseText = "";

      try {
        setLoading(true, "Running prediction...");
        let response;
        if (inputMode === "text") {
          const text = (caseText.value || "").trim();
          if (!text) {
            throw new Error("Please paste case text before running prediction.");
          }
          latestCaseText = text;
          response = await fetch("/api/predict/text", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              text,
              use_ollama: shouldUseOllama,
              ollama_model: requestedModel,
              requester_role: role
            })
          });
        } else {
          if (!caseFile.files || !caseFile.files.length) {
            throw new Error("Please select a PDF or DOCX file before running prediction.");
          }
          const formData = new FormData();
          formData.append("file", caseFile.files[0]);
          formData.append("use_ollama", shouldUseOllama ? "true" : "false");
          formData.append("ollama_model", requestedModel);
          formData.append("requester_role", role);
          response = await fetch("/api/predict/file", {
            method: "POST",
            body: formData
          });
        }

        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Prediction failed.");
        }

        renderPrediction(payload.prediction);
        if (!latestCaseText) {
          latestCaseText = (payload.prediction && payload.prediction.text_preview) ? payload.prediction.text_preview : "";
        }
        setLoading(false, "Prediction completed.");

        if (payload.ollama_explanation) {
          ollamaArea.classList.remove("hidden");
          ollamaText.textContent = payload.ollama_explanation;
        } else if (payload.ollama_error) {
          ollamaArea.classList.remove("hidden");
          ollamaText.textContent = "Ollama error: " + payload.ollama_error;
        }
      } catch (err) {
        setLoading(false, "Request failed.");
        showError(err.message || String(err));
      }
    }

    async function submitFeedback() {
      if (!latestPrediction) {
        feedbackStatus.textContent = "Run a prediction first, then save feedback.";
        return;
      }
      const actual = (actualOutcome.value || "").trim();
      if (!actual) {
        feedbackStatus.textContent = "Provide actual outcome before saving feedback.";
        return;
      }
      const notes = (feedbackNotes.value || "").trim();
      const role = (requesterRole.value || "regular_user").trim();
      const body = {
        case_text: latestCaseText || (latestPrediction.text_preview || ""),
        predicted_outcome: latestPrediction.predicted_outcome || "",
        actual_outcome: actual,
        requester_role: role,
        notes: notes
      };
      try {
        feedbackBtn.disabled = true;
        feedbackStatus.textContent = "Saving feedback...";
        const response = await fetch("/api/feedback", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || "Feedback save failed.");
        }
        feedbackStatus.textContent = "Feedback saved to: " + (payload.feedback_file || "file");
      } catch (err) {
        feedbackStatus.textContent = "Feedback error: " + (err.message || String(err));
      } finally {
        feedbackBtn.disabled = false;
      }
    }

    tabText.addEventListener("click", () => setMode("text"));
    tabFile.addEventListener("click", () => setMode("file"));
    requesterRole.addEventListener("change", updateRoleHelp);
    predictBtn.addEventListener("click", submitPrediction);
    feedbackBtn.addEventListener("click", submitFeedback);
    retrainBtn.addEventListener("click", startRetrainFromFeedback);
    updateRoleHelp();
    fetchRetrainStatus();
  </script>
</body>
</html>
"""

    return render_template_string(
        html,
        training_rows=training_rows,
        metadata=metadata,
        cv_results=cv_results,
        predictor_ready=predict_status["ready"],
        predictor_error=predict_status["error"],
        statute_status=statute_status,
        hybrid_status=predict_status.get("hybrid", {}),
        case_retriever_status=case_retriever_status,
        requester_roles=predict_status.get("requester_roles", {}),
        retrain_state=retrain_state,
        model_path=str(MODEL_PATH),
        default_ollama_model=OLLAMA_MODEL,
        ollama_models=ollama_models,
        ollama_error=ollama_error,
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
