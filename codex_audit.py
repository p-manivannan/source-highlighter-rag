"""Qualitatively audit a saved Ragas report with local Codex.

Run after ``evaluate_rag.py`` has produced ``evaluation_results.json``:

    uv run python codex_audit.py

The script invokes the Codex product through ``codex exec``. It does not call
Gemini, Ragas, Ollama, or the OpenAI API directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "evaluation_results.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "codex_audit_results.json"
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)
ROOT_CAUSES = (
    "retrieval_issue",
    "answer_generation_issue",
    "citation_grounding_issue",
    "evaluator_issue",
    "test_case_mismatch",
)
AUDIT_SCHEMA_VERSION = 1
CODEX_HELP_TIMEOUT_SECONDS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Existing Ragas evaluation report to audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination for the Codex audit report.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore completed audit records and re-audit every question.",
    )
    parser.add_argument(
        "--codex-bin",
        default=os.getenv("CODEX_BIN", "codex"),
        help="Codex executable name or absolute path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600,
        help="Timeout for each per-question Codex audit.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Maximum total retrieved-context characters included per prompt.",
    )
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run evaluate_rag.py first.")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def windows_codex_candidates() -> list[Path]:
    """Return common packaged-app Codex locations on Windows."""
    candidates: list[Path] = []
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
        try:
            candidates.extend(root.glob("*/codex.exe"))
            candidates.extend(root.glob("*/codex"))
        except OSError:
            pass

    for root in (
        Path("C:/Program Files/WindowsApps"),
        Path("D:/Program Files/WindowsApps"),
        Path("D:/WindowsApps"),
    ):
        try:
            candidates.extend(root.glob("OpenAI.Codex_*/*/resources/codex.exe"))
            candidates.extend(root.glob("OpenAI.Codex_*/app/resources/codex.exe"))
        except OSError:
            continue
    return sorted(
        candidates,
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def powershell_codex_command() -> str | None:
    """Ask PowerShell where Codex lives when Python's PATH view is stale."""
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Command codex -ErrorAction SilentlyContinue).Source",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    resolved = completed.stdout.strip()
    return resolved if completed.returncode == 0 and resolved else None


def codex_supports_audit_exec(codex_bin: str) -> bool:
    """Return whether an executable looks like the Codex CLI used for audits."""
    try:
        completed = subprocess.run(
            [codex_bin, "exec", "--help"],
            text=True,
            capture_output=True,
            timeout=CODEX_HELP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    output = f"{completed.stdout}\n{completed.stderr}".casefold()
    return (
        completed.returncode == 0
        and "exec" in output
        and "--output-schema" in output
        and "--sandbox" in output
    )


def resolve_codex_bin(codex_bin: str) -> str | None:
    """Resolve a Codex CLI binary that supports ``codex exec`` audit mode."""
    configured = os.getenv("CODEX_BIN", "").strip()
    requested = configured or codex_bin
    expanded = Path(os.path.expandvars(os.path.expanduser(requested)))
    if expanded.is_file() and codex_supports_audit_exec(str(expanded)):
        return str(expanded)

    for name in (requested, f"{requested}.exe", f"{requested}.cmd"):
        resolved = shutil.which(name)
        if (
            resolved
            and "windowsapps" not in resolved.casefold()
            and codex_supports_audit_exec(resolved)
        ):
            return resolved

    powershell_resolved = powershell_codex_command()
    if (
        powershell_resolved
        and Path(powershell_resolved).is_file()
        and "windowsapps" not in powershell_resolved.casefold()
        and codex_supports_audit_exec(powershell_resolved)
    ):
        return powershell_resolved

    for candidate in windows_codex_candidates():
        if (
            candidate.is_file()
            and "windowsapps" not in str(candidate).casefold()
            and codex_supports_audit_exec(str(candidate))
        ):
            return str(candidate)
    return None


def codex_is_available(codex_bin: str) -> bool:
    return resolve_codex_bin(codex_bin) is not None


def audit_output_schema() -> dict[str, Any]:
    metric_score = {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }
    metric_rationale = {
        "type": "string",
        "minLength": 1,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {name: metric_score for name in METRIC_NAMES},
                "required": list(METRIC_NAMES),
            },
            "rationales": {
                "type": "object",
                "additionalProperties": False,
                "properties": {name: metric_rationale for name in METRIC_NAMES},
                "required": list(METRIC_NAMES),
            },
            "agrees_with_ragas": {"type": "boolean"},
            "disagreement_notes": {"type": "string"},
            "likely_root_causes": {
                "type": "array",
                "items": {"type": "string", "enum": list(ROOT_CAUSES)},
                "minItems": 1,
            },
            "recommended_fixes": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "overall_summary": {"type": "string", "minLength": 1},
        },
        "required": [
            "scores",
            "rationales",
            "agrees_with_ragas",
            "disagreement_notes",
            "likely_root_causes",
            "recommended_fixes",
            "overall_summary",
        ],
    }


def bounded_contexts(contexts: list[Any], max_context_chars: int) -> list[str]:
    remaining = max(0, max_context_chars)
    bounded: list[str] = []
    for context in contexts:
        text = str(context)
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n[TRUNCATED]"
        bounded.append(text)
        remaining -= len(text)
    return bounded


def build_audit_prompt(sample: dict[str, Any], max_context_chars: int) -> str:
    payload = {
        "question_number": sample.get("question_number"),
        "question": sample.get("question", ""),
        "generated_answer": sample.get("answer", ""),
        "retrieved_contexts": bounded_contexts(
            list(sample.get("contexts", [])),
            max_context_chars,
        ),
        "ground_truth_answer_points": sample.get("ground_truth", ""),
        "expected_sources": sample.get("expected_sources", []),
        "retrieved_sources": sample.get("retrieved_sources", []),
        "retrieval_diagnostics": sample.get("retrieval_diagnostics", {}),
        "ragas_scores": sample.get("scores", {}),
        "ragas_failure_patterns": sample.get("failure_patterns", []),
    }
    return (
        "You are auditing one RAG evaluation sample. Judge only from the JSON "
        "payload below; do not use outside knowledge.\n\n"
        "Use these Ragas-style dimensions:\n"
        "- context_precision: retrieved contexts are relevant to the question and reference.\n"
        "- context_recall: retrieved contexts contain enough evidence for the reference answer.\n"
        "- faithfulness: the generated answer is supported by retrieved contexts.\n"
        "- answer_relevancy: the generated answer directly addresses the user question.\n"
        "- answer_correctness: the generated answer matches the reference answer points.\n\n"
        "Return JSON that conforms exactly to the provided schema. Scores must be "
        "numbers from 0 to 1. Keep each rationale and fix short, concrete, and "
        "specific to this sample. Decide whether the existing Ragas/Qwen score "
        "pattern looks broadly reliable; set agrees_with_ragas=false when the "
        "saved scores appear inconsistent with the visible evidence.\n\n"
        f"SAMPLE_JSON:\n{json.dumps(payload, indent=2, ensure_ascii=True)}"
    )


def validate_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Codex audit output must be a JSON object.")
    scores = payload.get("scores")
    rationales = payload.get("rationales")
    if not isinstance(scores, dict) or not isinstance(rationales, dict):
        raise ValueError("Codex audit output is missing scores or rationales.")
    normalized_scores: dict[str, float] = {}
    normalized_rationales: dict[str, str] = {}
    for metric_name in METRIC_NAMES:
        try:
            score = float(scores[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing numeric score for {metric_name}.") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score for {metric_name} must be between 0 and 1.")
        rationale = str(rationales.get(metric_name, "")).strip()
        if not rationale:
            raise ValueError(f"Missing rationale for {metric_name}.")
        normalized_scores[metric_name] = score
        normalized_rationales[metric_name] = rationale

    root_causes = payload.get("likely_root_causes")
    if not isinstance(root_causes, list) or not root_causes:
        raise ValueError("Codex audit output must include likely_root_causes.")
    invalid_causes = [cause for cause in root_causes if cause not in ROOT_CAUSES]
    if invalid_causes:
        raise ValueError(f"Invalid root causes: {invalid_causes}")

    fixes = payload.get("recommended_fixes")
    if not isinstance(fixes, list) or not any(str(fix).strip() for fix in fixes):
        raise ValueError("Codex audit output must include recommended_fixes.")

    return {
        "scores": normalized_scores,
        "rationales": normalized_rationales,
        "agrees_with_ragas": bool(payload.get("agrees_with_ragas")),
        "disagreement_notes": str(payload.get("disagreement_notes", "")).strip(),
        "likely_root_causes": [str(cause) for cause in root_causes],
        "recommended_fixes": [
            str(fix).strip() for fix in fixes if str(fix).strip()
        ],
        "overall_summary": str(payload.get("overall_summary", "")).strip(),
    }


def parse_codex_json_output(stdout: str) -> dict[str, Any]:
    """Parse the JSON object from Codex stdout, tolerating launcher noise."""
    decoder = json.JSONDecoder()
    stripped = stdout.strip()
    if not stripped:
        raise ValueError("Codex returned empty stdout.")

    starts = [index for index, char in enumerate(stripped) if char == "{"]
    for start in starts:
        try:
            payload, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if not stripped[start + end :].strip():
            if not isinstance(payload, dict):
                raise ValueError("Codex audit output must be a JSON object.")
            return payload

    preview = stripped if len(stripped) <= 500 else f"{stripped[:500]}..."
    raise ValueError(f"Codex returned no JSON object on stdout: {preview}")


def run_codex_audit(
    *,
    codex_bin: str,
    prompt: str,
    schema_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    resolved_codex = resolve_codex_bin(codex_bin)
    if resolved_codex is None:
        raise FileNotFoundError(
            "Codex CLI with `exec --output-schema` support was not found: "
            f"{codex_bin}"
        )

    command = [
        resolved_codex,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        prompt,
    ]
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"codex exec exited with code {completed.returncode}: {stderr}"
        )
    payload = parse_codex_json_output(completed.stdout)
    return validate_audit_payload(payload)


def load_existing_audits(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    records = payload.get("per_question")
    return records if isinstance(records, list) else []


def completed_audit_lookup(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("audit_error"):
            continue
        try:
            question_number = int(record["question_number"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(record.get("scores"), dict):
            lookup[question_number] = record
    return lookup


def aggregate_scores(per_question: list[dict[str, Any]]) -> dict[str, float | None]:
    aggregates: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        values = [
            float(item["scores"][metric_name])
            for item in per_question
            if not item.get("audit_error")
            and isinstance(item.get("scores"), dict)
            and item["scores"].get(metric_name) is not None
        ]
        aggregates[metric_name] = fmean(values) if values else None
    return aggregates


def disagreement_summary(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    audited = [
        item
        for item in per_question
        if not item.get("audit_error") and "agrees_with_ragas" in item
    ]
    disagreement_count = sum(
        1 for item in audited if item.get("agrees_with_ragas") is False
    )
    return {
        "audited_question_count": len(audited),
        "ragas_disagreement_count": disagreement_count,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    temporary_path.replace(path)


def build_report(
    *,
    source_report_path: Path,
    output_path: Path,
    codex_command: list[str],
    per_question: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audited_at": datetime.now(UTC).isoformat(),
            "source_report_path": str(source_report_path),
            "codex_command": codex_command,
            "question_count": len(per_question),
        },
        "per_question": per_question,
        "aggregate_scores": aggregate_scores(per_question),
        "disagreement_summary": disagreement_summary(per_question),
    }


def run_audit(
    *,
    input_path: Path,
    output_path: Path,
    refresh: bool = False,
    codex_bin: str = "codex",
    timeout_seconds: float = 600,
    max_context_chars: int = 12000,
) -> dict[str, Any]:
    resolved_codex = resolve_codex_bin(codex_bin)
    if resolved_codex is None:
        raise FileNotFoundError(
            "Codex CLI with `exec --output-schema` support was not found: "
            f"{codex_bin}"
        )

    source_report = load_json_object(input_path)
    samples = source_report.get("per_question")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{input_path} contains no per_question records.")

    existing = [] if refresh else load_existing_audits(output_path)
    completed = completed_audit_lookup(existing)
    per_question: list[dict[str, Any]] = []
    codex_command = [
        resolved_codex,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        "<runtime-schema-path>",
        "<per-question-prompt>",
    ]

    with tempfile.TemporaryDirectory() as temporary_directory:
        schema_path = Path(temporary_directory) / "codex_audit_schema.json"
        schema_path.write_text(
            json.dumps(audit_output_schema(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        for index, sample in enumerate(samples, start=1):
            question_number = int(sample.get("question_number", index))
            cached = completed.get(question_number)
            if cached is not None:
                per_question.append(cached)
                print(f"Reused Codex audit {question_number}/{len(samples)}")
                continue

            base_record = {
                "question_number": question_number,
                "question": str(sample.get("question", "")),
            }
            try:
                prompt = build_audit_prompt(sample, max_context_chars)
                audit = run_codex_audit(
                    codex_bin=resolved_codex,
                    prompt=prompt,
                    schema_path=schema_path,
                    timeout_seconds=timeout_seconds,
                )
                record = {**base_record, **audit}
                print(f"Audited {question_number}/{len(samples)}")
            except Exception as exc:
                record = {**base_record, "audit_error": str(exc)}
                print(f"Audit failed {question_number}/{len(samples)}: {exc}")

            per_question.append(record)
            partial_report = build_report(
                source_report_path=input_path,
                output_path=output_path,
                codex_command=codex_command,
                per_question=per_question,
            )
            write_report(output_path, partial_report)

    report = build_report(
        source_report_path=input_path,
        output_path=output_path,
        codex_command=codex_command,
        per_question=per_question,
    )
    write_report(output_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = run_audit(
        input_path=args.input,
        output_path=args.output,
        refresh=args.refresh,
        codex_bin=args.codex_bin,
        timeout_seconds=args.timeout_seconds,
        max_context_chars=args.max_context_chars,
    )
    audited = report["disagreement_summary"]["audited_question_count"]
    disagreements = report["disagreement_summary"]["ragas_disagreement_count"]
    print(
        "Codex audit complete: "
        f"{audited} audited question(s), {disagreements} Ragas disagreement(s)."
    )
    print(f"Saved Codex audit report to {args.output}")


if __name__ == "__main__":
    main()
