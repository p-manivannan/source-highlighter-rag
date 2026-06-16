"""Evaluate the citation-grounded RAG pipeline with the Ragas triad metrics.

Run from the project directory after ingestion:

    uv run python evaluate_rag.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings.base import LlamaIndexEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_DIR = Path("/workspace")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_EVALUATION_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)
FAILURE_THRESHOLD = 0.7

FAILURE_EXPLANATIONS = {
    "context_precision": (
        "Low context precision: retrieval includes distracting or irrelevant chunks."
    ),
    "context_recall": (
        "Low context recall: retrieval is missing information needed for the "
        "reference answer."
    ),
    "faithfulness": (
        "Low faithfulness: generated claims are not sufficiently grounded in the "
        "retrieved context."
    ),
    "answer_relevancy": (
        "Low answer relevancy: the answer does not directly address the question."
    ),
    "answer_correctness": (
        "Low answer correctness: the answer conflicts with or omits important "
        "ground-truth points."
    ),
    "expected_source_coverage": (
        "Missing expected sources: retrieval did not return all documents identified "
        "by the test case."
    ),
    "metric_unavailable": (
        "Metric unavailable: the Ragas scoring job failed or timed out for this question."
    ),
}
ProgressCallback = Callable[[str, float], None]


def normalize_ollama_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def ollama_model_names(payload: dict[str, Any]) -> set[str]:
    """Return all model identifiers exposed by Ollama's tags endpoint."""
    names = set()
    for model in payload.get("models", []):
        if not isinstance(model, dict):
            continue
        for field in ("name", "model"):
            value = str(model.get(field, "")).strip()
            if value:
                names.add(value)
    return names


def check_ollama(
    *,
    base_url: str,
    model_name: str,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """Verify that Ollama is reachable and the configured judge is installed."""
    normalized_url = normalize_ollama_base_url(base_url)
    try:
        with urllib.request.urlopen(
            f"{normalized_url}/api/tags",
            timeout=timeout,
        ) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return (
            False,
            f"Could not reach Ollama at {normalized_url}: {exc}",
        )

    installed_models = ollama_model_names(payload)
    if model_name not in installed_models:
        return (
            False,
            f"Ollama model '{model_name}' is not installed. "
            f"Run `ollama pull {model_name}`.",
        )
    return True, ""


def create_ragas_judge(
    *,
    provider: str,
    ollama_model: str,
    ollama_base_url: str,
):
    """Create the local judge used by every LLM-based Ragas metric."""
    if provider.casefold() != "ollama":
        raise ValueError(
            "EVALUATION_LLM_PROVIDER must be 'ollama' for quota-safe evaluation."
        )
    available, error = check_ollama(
        base_url=ollama_base_url,
        model_name=ollama_model,
    )
    if not available:
        raise RuntimeError(error)

    local_llm = ChatOllama(
        model=ollama_model,
        base_url=normalize_ollama_base_url(ollama_base_url),
        temperature=0.0,
        format="json",
        num_ctx=8192,
    )
    return LangchainLLMWrapper(local_llm)


def workspace_path(relative_path: str) -> Path:
    """Use /workspace in deployment and the repository root for local runs."""
    workspace_dir = Path(os.getenv("WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR))
    if not workspace_dir.exists():
        workspace_dir = BASE_DIR
    return workspace_dir / relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=workspace_path("data/demo_questions.json"),
        help="JSON file containing question, expected_sources, and answer_points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_path("evaluation_results.json"),
        help="Destination for the complete evaluation report.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=FAILURE_THRESHOLD,
        help="Scores below this value are classified as failure patterns.",
    )
    return parser.parse_args()


def load_test_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path} must contain a non-empty JSON array.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"Question {index} is missing a non-empty 'question'.")
        if not isinstance(item.get("answer_points"), list):
            raise ValueError(f"Question {index} is missing an 'answer_points' list.")
    return questions


def concatenate_answer_points(answer_points: list[Any]) -> str:
    """Build the Ragas ground truth from the test case's expected answer points."""
    points = [str(point).strip().rstrip(".") for point in answer_points if str(point).strip()]
    return ". ".join(points) + ("." if points else "")


def extract_answer_text(answer_payload: dict[str, Any]) -> str:
    """Extract only claims that survived GroundedAnswer evidence validation."""
    if answer_payload.get("format") == "structured":
        claims = answer_payload.get("claims") or []
        validated_text = " ".join(
            str(claim.get("text", "")).strip()
            for claim in claims
            if str(claim.get("text", "")).strip()
        )
        if validated_text:
            return validated_text
    return str(answer_payload.get("content", "")).strip()


def finite_score(value: Any) -> float | None:
    """Convert Ragas/numpy values to JSON-safe floats."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def expected_source_coverage(
    expected_sources: list[Any],
    retrieved_sources: list[str],
) -> tuple[float | None, list[str]]:
    expected_lookup = {
        str(source).strip().casefold(): str(source).strip()
        for source in expected_sources
        if str(source).strip()
    }
    retrieved = {source.strip().casefold() for source in retrieved_sources}
    if not expected_lookup:
        return None, []
    missing = sorted(
        original
        for normalized, original in expected_lookup.items()
        if normalized not in retrieved
    )
    matched_count = sum(normalized in retrieved for normalized in expected_lookup)
    return matched_count / len(expected_lookup), missing


def classify_failures(
    scores: dict[str, float | None],
    missing_sources: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    failures = []
    for metric_name in METRIC_NAMES:
        score = scores.get(metric_name)
        if score is None:
            failures.append(
                {
                    "pattern": "metric_unavailable",
                    "metric": metric_name,
                    "explanation": (
                        f"{FAILURE_EXPLANATIONS['metric_unavailable']} "
                        f"Metric: {metric_name}."
                    ),
                }
            )
        elif score < threshold:
            failures.append(
                {
                    "pattern": metric_name,
                    "score": score,
                    "explanation": FAILURE_EXPLANATIONS[metric_name],
                }
            )
    if missing_sources:
        failures.append(
            {
                "pattern": "expected_source_coverage",
                "missing_sources": missing_sources,
                "explanation": FAILURE_EXPLANATIONS["expected_source_coverage"],
            }
        )
    return failures


def aggregate_scores(per_question: list[dict[str, Any]]) -> dict[str, float | None]:
    aggregates: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        values = [
            item["scores"][metric_name]
            for item in per_question
            if item["scores"].get(metric_name) is not None
        ]
        aggregates[metric_name] = fmean(values) if values else None

    coverage_values = [
        item["retrieval_diagnostics"]["expected_source_coverage"]
        for item in per_question
        if item["retrieval_diagnostics"]["expected_source_coverage"] is not None
    ]
    aggregates["expected_source_coverage"] = (
        fmean(coverage_values) if coverage_values else None
    )
    return aggregates


def validate_metric_scores(per_question: list[dict[str, Any]]) -> None:
    """Reject reports whose Ragas jobs silently failed into NaN values."""
    missing_metrics = [
        metric_name
        for metric_name in METRIC_NAMES
        if not any(
            item["scores"].get(metric_name) is not None for item in per_question
        )
    ]
    if missing_metrics:
        rendered = ", ".join(missing_metrics)
        raise RuntimeError(
            "Ragas returned no valid scores for: "
            f"{rendered}. Check the evaluator log for provider errors."
        )


def run_evaluation(
    *,
    api_key: str,
    model_name: str,
    questions_path: Path,
    output_path: Path,
    failure_threshold: float = FAILURE_THRESHOLD,
    progress_callback: ProgressCallback | None = None,
    engine: Any | None = None,
    evaluation_provider: str | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    """Run the complete Ragas evaluation and return the saved report."""
    if not 0.0 <= failure_threshold <= 1.0:
        raise ValueError("failure_threshold must be between 0 and 1.")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    evaluation_provider = (
        evaluation_provider
        or os.getenv(
            "EVALUATION_LLM_PROVIDER",
            DEFAULT_EVALUATION_LLM_PROVIDER,
        )
    ).strip()
    ollama_model = (
        ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    ).strip()
    ollama_base_url = (
        ollama_base_url
        or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    ).strip()

    questions = load_test_questions(questions_path)
    if engine is None:
        from app import load_engine

        engine = load_engine(api_key, model_name)

    samples: list[SingleTurnSample] = []
    sample_records: list[dict[str, Any]] = []
    for index, test_case in enumerate(questions, start=1):
        question = str(test_case["question"]).strip()
        answer_payload, retrieved_chunks = engine.answer(
            question,
            allow_legacy_fallback=False,
        )
        answer = extract_answer_text(answer_payload)
        contexts = [chunk.text for chunk in retrieved_chunks]
        ground_truth = concatenate_answer_points(test_case["answer_points"])
        if not ground_truth:
            raise ValueError(
                f"Question {index} has no usable answer_points; reference-based "
                "Ragas metrics cannot be calculated."
            )

        retrieved_sources = list(dict.fromkeys(chunk.source_file for chunk in retrieved_chunks))
        coverage, missing_sources = expected_source_coverage(
            test_case.get("expected_sources", []),
            retrieved_sources,
        )
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        sample_records.append(
            {
                "question_number": index,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "expected_sources": test_case.get("expected_sources", []),
                "retrieved_sources": retrieved_sources,
                "retrieval_diagnostics": {
                    "expected_source_coverage": coverage,
                    "missing_expected_sources": missing_sources,
                },
            }
        )
        if progress_callback is not None:
            progress_callback(
                f"Generated answer {index} of {len(questions)}",
                0.7 * index / len(questions),
            )
        print(f"Prepared {index}/{len(questions)}: {question}")

    if progress_callback is not None:
        progress_callback("Calculating Ragas metrics", 0.75)
    ragas_llm = create_ragas_judge(
        provider=evaluation_provider,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )
    ragas_embeddings = LlamaIndexEmbeddingsWrapper(
        engine.vector_retriever._embed_model
    )
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            timeout=600,
            max_retries=3,
            max_wait=20,
            max_workers=4,
        ),
        raise_exceptions=False,
    )
    score_rows = result.to_pandas().to_dict(orient="records")

    per_question = []
    failure_counts: Counter[str] = Counter()
    for record, score_row in zip(sample_records, score_rows, strict=True):
        scores = {
            metric_name: finite_score(score_row.get(metric_name))
            for metric_name in METRIC_NAMES
        }
        failures = classify_failures(
            scores,
            record["retrieval_diagnostics"]["missing_expected_sources"],
            failure_threshold,
        )
        failure_counts.update({failure["pattern"] for failure in failures})
        per_question.append({**record, "scores": scores, "failure_patterns": failures})
        rendered_scores = ", ".join(
            f"{name}={'n/a' if score is None else f'{score:.3f}'}"
            for name, score in scores.items()
        )
        print(f"Question {record['question_number']} scores: {rendered_scores}")

    validate_metric_scores(per_question)
    report = {
        "metadata": {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "rag_framework": "Ragas RAG triad",
            "gemini_model": model_name,
            "evaluation_llm_provider": evaluation_provider,
            "evaluation_model": ollama_model,
            "ollama_base_url": normalize_ollama_base_url(ollama_base_url),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "failure_threshold": failure_threshold,
            "question_count": len(per_question),
        },
        "per_question": per_question,
        "aggregate_scores": aggregate_scores(per_question),
        "failure_pattern_summary": {
            pattern: {
                "question_count": count,
                "explanation": FAILURE_EXPLANATIONS.get(
                    pattern,
                    "One or more evaluation checks failed.",
                ),
            }
            for pattern, count in failure_counts.most_common()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if progress_callback is not None:
        progress_callback("Evaluation complete", 1.0)
    return report


def main() -> None:
    args = parse_args()
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from the environment or .env.")

    model_name = os.getenv("GOOGLE_MODEL", "models/gemini-2.5-flash").strip()
    report = run_evaluation(
        api_key=api_key,
        model_name=model_name,
        questions_path=args.questions,
        output_path=args.output,
        failure_threshold=args.failure_threshold,
    )

    print("\nAggregate scores:")
    for metric_name, score in report["aggregate_scores"].items():
        rendered = "n/a" if score is None else f"{score:.3f}"
        print(f"  {metric_name}: {rendered}")
    print(f"\nSaved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
