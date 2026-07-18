"""
Utilities for loading structured CSV/JSON case data.
"""
from pathlib import Path
import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)


TEXT_COLUMNS = ["facts", "issues", "arguments", "analysis", "decision"]
LABEL_ALIASES = ["label", "target", "outcome_label", "case_outcome"]
OUTCOME_ALIASES = ["outcome", *LABEL_ALIASES]
FULL_TEXT_ALIASES = ["full_text", "text", "judgment_text", "case_text", "body"]
SECTION_ALIASES = {
    "facts": ["facts", "fact", "background"],
    "issues": ["issues", "issue", "questions_for_determination"],
    "arguments": ["arguments", "argument", "submissions", "counsel_arguments"],
    "analysis": ["analysis", "reasoning", "ratio", "holding"],
    "decision": ["decision", "judgment", "judgement", "ruling", "order"],
}
METADATA_ALIASES = {
    "court": ["court", "court_name", "tribunal"],
    "year": ["year", "judgment_year", "judgement_year", "decision_year"],
}
OUTCOME_SUCCESS_PATTERNS = [
    r"\bplaintiff[_\s]+success\b",
    r"\bgranted\b",
    r"\bappeal\s+allowed\b",
    r"\bappeal\s+succeeds?\b",
    r"\bapplication\s+granted\b",
    r"\bconviction\s+quashed\b",
    r"\bacquittal\b",
    r"\bdeclaration\s+granted\b",
    r"\bplaintiff\s+succeeds?\b",
]
OUTCOME_FAILURE_PATTERNS = [
    r"\bplaintiff[_\s]+failure\b",
    r"\bdismissed\b",
    r"\bappeal\s+dismissed\b",
    r"\bappeal\s+fails?\b",
    r"\bconviction\s+upheld\b",
    r"\bclaim\s+dismissed\b",
    r"\bplaintiff\s+fails?\b",
    r"\brelief\s+denied\b",
]
OUTCOME_REMITTED_PATTERNS = [
    r"\bremitted\b",
    r"\bretrial\b",
    r"\bre-hearing\b",
]


def _normalize_name(name):
    """Normalize column names to improve alias matching."""
    lowered = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_.]", "", lowered)


def _find_alias_column(columns, aliases):
    """Return first matching column, accepting pandas duplicate suffixes (e.g. .1)."""
    normalized = {_normalize_name(col): col for col in columns}

    # Exact matches first
    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]

    # Then duplicate suffix matches (e.g., outcome.1)
    for alias in aliases:
        pattern = re.compile(rf"^{re.escape(_normalize_name(alias))}\.\d+$")
        for col_norm, original_col in normalized.items():
            if pattern.match(col_norm):
                return original_col
    return None


def _coalesce_columns(df, aliases):
    """Combine alias columns into a single series using first non-empty value per row."""
    candidates = []
    for alias in aliases:
        alias_norm = _normalize_name(alias)
        for col in df.columns:
            col_norm = _normalize_name(col)
            if col_norm == alias_norm or re.match(rf"^{re.escape(alias_norm)}\.\d+$", col_norm):
                candidates.append(col)

    if not candidates:
        return None

    # Keep order stable and unique
    seen = set()
    ordered = []
    for col in candidates:
        if col not in seen:
            ordered.append(col)
            seen.add(col)

    series = df[ordered[0]]
    for col in ordered[1:]:
        series = series.where(series.notna() & (series.astype(str).str.strip() != ""), df[col])
    return series


def _read_json_flex(path):
    """
    Read JSON with flexible support for JSON Lines or a JSON array.
    """
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path, lines=False)


def canonicalize_outcome_label(label):
    """
    Map raw outcome labels to a stable land-focused label set:
    granted | dismissed | matter remitted | other
    """
    if label is None:
        return ""

    normalized = str(label).strip().lower()
    if normalized in {"", "nan", "none", "null", "<na>", "n/a", "na"}:
        return ""
    if normalized in {"other", "unclear"}:
        return "other"

    for pattern in OUTCOME_REMITTED_PATTERNS:
        if re.search(pattern, normalized):
            return "matter remitted"
    for pattern in OUTCOME_SUCCESS_PATTERNS:
        if re.search(pattern, normalized):
            return "granted"
    for pattern in OUTCOME_FAILURE_PATTERNS:
        if re.search(pattern, normalized):
            return "dismissed"

    return "other"


def load_structured_cases(file_path):
    """
    Load cases from CSV/JSON into a normalized DataFrame.

    Required:
      - outcome (or alias: label/target/outcome_label/case_outcome)
      - full_text (or text, or derived from section columns)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = _read_json_flex(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .json")

    if df.empty:
        raise ValueError("Structured data file is empty.")

    # Normalize whitespace in raw column names
    df.columns = [str(c).strip() for c in df.columns]

    # Map label aliases and duplicate columns -> outcome
    outcome_series = _coalesce_columns(df, OUTCOME_ALIASES)
    if outcome_series is not None:
        df["outcome"] = outcome_series

    if "outcome" not in df.columns:
        raise ValueError(
            "Missing required label column 'outcome'. "
            "Add 'outcome' or one of: label, target, outcome_label, case_outcome."
        )

    # Map section aliases to canonical names
    for canonical, aliases in SECTION_ALIASES.items():
        matched_col = _find_alias_column(df.columns, aliases)
        if matched_col and canonical not in df.columns:
            df[canonical] = df[matched_col]

    # Map metadata aliases
    for canonical, aliases in METADATA_ALIASES.items():
        matched_col = _find_alias_column(df.columns, aliases)
        if matched_col and canonical not in df.columns:
            df[canonical] = df[matched_col]

    # Build full_text if missing
    if "full_text" not in df.columns:
        full_text_col = _find_alias_column(df.columns, FULL_TEXT_ALIASES)
        if full_text_col:
            df["full_text"] = df[full_text_col]
        else:
            available_sections = [c for c in TEXT_COLUMNS if c in df.columns]
            if not available_sections:
                # Fallback: combine all object-like feature columns into one text field
                fallback_cols = [
                    c for c in df.columns
                    if c != "outcome" and df[c].dtype == "object"
                ]
                if not fallback_cols:
                    raise ValueError(
                        "Missing required text column. Provide 'full_text' or 'text', "
                        "or at least one of: facts, issues, arguments, analysis, decision."
                    )
                df["full_text"] = df[fallback_cols].fillna("").agg(" ".join, axis=1)
            else:
                df["full_text"] = df[available_sections].fillna("").agg("\n\n".join, axis=1)

    # Ensure section columns exist
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Clean outcomes
    outcome_series = df["outcome"].where(df["outcome"].notna(), "")
    outcome_series = (
        outcome_series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    outcome_series = outcome_series.replace(
        {
            "nan": "",
            "none": "",
            "null": "",
            "<na>": "",
            "n/a": "",
            "na": "",
        }
    )
    df["outcome"] = outcome_series.apply(canonicalize_outcome_label)
    before = len(df)
    df = df[(df["outcome"].notna()) & (df["outcome"] != "")]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing outcome labels.")

    # Normalize year if present
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def load_and_merge_structured_cases(file_paths):
    """
    Load and merge multiple CSV/JSON files into one normalized DataFrame.
    """
    if not file_paths:
        raise ValueError("No structured input files provided.")

    frames = []
    for file_path in file_paths:
        frame = load_structured_cases(file_path)
        frame["source_file"] = str(file_path)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if merged.empty:
        raise ValueError("No rows available after merging structured files.")

    logger.info(
        "Merged %s structured files into %s rows",
        len(file_paths),
        len(merged),
    )
    return merged
