import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import codex_audit


def sample_record(question_number: int = 1) -> dict:
    return {
        "question_number": question_number,
        "question": "What failed?",
        "answer": "The patch process failed.",
        "contexts": [
            "Patch notices were sent, but the vulnerable system was not patched.",
            "Unrelated trailing context.",
        ],
        "ground_truth": "known vulnerability. patch was available.",
        "expected_sources": ["source.pdf"],
        "retrieved_sources": ["source.pdf"],
        "retrieval_diagnostics": {
            "expected_source_coverage": 1.0,
            "missing_expected_sources": [],
        },
        "scores": {
            "context_precision": 0.8,
            "context_recall": 0.9,
            "faithfulness": 0.4,
            "answer_relevancy": 0.7,
            "answer_correctness": 0.6,
        },
        "failure_patterns": [{"pattern": "faithfulness"}],
    }


def audit_payload() -> dict:
    return {
        "scores": {metric_name: 0.75 for metric_name in codex_audit.METRIC_NAMES},
        "rationales": {
            metric_name: f"{metric_name} rationale"
            for metric_name in codex_audit.METRIC_NAMES
        },
        "agrees_with_ragas": False,
        "disagreement_notes": "The saved faithfulness score looks too low.",
        "likely_root_causes": ["evaluator_issue"],
        "recommended_fixes": ["Review the Ragas judge model."],
        "overall_summary": "The answer is mostly grounded.",
    }


class CodexAuditPromptTests(unittest.TestCase):
    def test_build_audit_prompt_includes_metric_framework_and_bounded_context(self) -> None:
        sample = sample_record()
        prompt = codex_audit.build_audit_prompt(sample, max_context_chars=40)

        self.assertIn("context_precision", prompt)
        self.assertIn("faithfulness", prompt)
        self.assertIn("SAMPLE_JSON", prompt)
        self.assertIn("[TRUNCATED]", prompt)
        self.assertIn("What failed?", prompt)


class CodexAuditValidationTests(unittest.TestCase):
    def test_audit_output_schema_uses_supported_response_format_keywords(self) -> None:
        schema = codex_audit.audit_output_schema()

        root_causes = schema["properties"]["likely_root_causes"]
        self.assertNotIn("uniqueItems", root_causes)

    def test_validate_audit_payload_normalizes_scores(self) -> None:
        payload = audit_payload()
        payload["scores"]["faithfulness"] = "0.5"

        validated = codex_audit.validate_audit_payload(payload)

        self.assertEqual(0.5, validated["scores"]["faithfulness"])
        self.assertEqual(["evaluator_issue"], validated["likely_root_causes"])

    def test_validate_audit_payload_rejects_invalid_score(self) -> None:
        payload = audit_payload()
        payload["scores"]["faithfulness"] = 1.5

        with self.assertRaises(ValueError):
            codex_audit.validate_audit_payload(payload)


class CodexResolutionTests(unittest.TestCase):
    def test_resolve_codex_bin_prefers_code_bin_env_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            codex_path = Path(temporary_directory) / "codex.exe"
            codex_path.write_text("", encoding="utf-8")

            with (
                patch.dict("codex_audit.os.environ", {"CODEX_BIN": str(codex_path)}),
                patch("codex_audit.codex_supports_audit_exec", return_value=True),
            ):
                self.assertEqual(str(codex_path), codex_audit.resolve_codex_bin("codex"))

    def test_resolve_codex_bin_falls_back_to_powershell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            codex_path = Path(temporary_directory) / "codex.exe"
            codex_path.write_text("", encoding="utf-8")

            with (
                patch.dict("codex_audit.os.environ", {}, clear=True),
                patch("codex_audit.shutil.which", return_value=None),
                patch("codex_audit.powershell_codex_command", return_value=str(codex_path)),
                patch("codex_audit.codex_supports_audit_exec", return_value=True),
            ):
                self.assertEqual(str(codex_path), codex_audit.resolve_codex_bin("codex"))

    def test_resolve_codex_bin_rejects_non_exec_desktop_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            codex_path = Path(temporary_directory) / "codex.exe"
            codex_path.write_text("", encoding="utf-8")

            with (
                patch.dict("codex_audit.os.environ", {"CODEX_BIN": str(codex_path)}),
                patch("codex_audit.shutil.which", return_value=None),
                patch("codex_audit.powershell_codex_command", return_value=None),
                patch("codex_audit.windows_codex_candidates", return_value=[]),
                patch("codex_audit.codex_supports_audit_exec", return_value=False),
            ):
                self.assertIsNone(codex_audit.resolve_codex_bin("codex"))

    def test_windows_codex_candidates_include_local_app_data_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            codex_path = root / "OpenAI" / "Codex" / "bin" / "abc123" / "codex.exe"
            codex_path.parent.mkdir(parents=True)
            codex_path.write_text("", encoding="utf-8")

            with patch.dict("codex_audit.os.environ", {"LOCALAPPDATA": str(root)}):
                candidates = codex_audit.windows_codex_candidates()

            self.assertIn(codex_path, candidates)


class CodexJsonParsingTests(unittest.TestCase):
    def test_parse_codex_json_output_allows_launcher_noise_prefix(self) -> None:
        payload = audit_payload()
        parsed = codex_audit.parse_codex_json_output(
            "Opening in existing browser session.\n" + json.dumps(payload)
        )

        self.assertEqual(payload["overall_summary"], parsed["overall_summary"])

    def test_parse_codex_json_output_rejects_output_without_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "no JSON object"):
            codex_audit.parse_codex_json_output("Opening in existing browser session.\n")


class CodexAuditRunTests(unittest.TestCase):
    def test_run_audit_requires_existing_input_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch("codex_audit.resolve_codex_bin", return_value="codex"),
                self.assertRaises(FileNotFoundError),
            ):
                codex_audit.run_audit(
                    input_path=root / "missing.json",
                    output_path=root / "audit.json",
                )

    def test_run_audit_requires_codex_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "evaluation_results.json"
            input_path.write_text(
                json.dumps({"per_question": [sample_record()]}),
                encoding="utf-8",
            )

            with (
                patch("codex_audit.resolve_codex_bin", return_value=None),
                self.assertRaises(FileNotFoundError),
            ):
                codex_audit.run_audit(
                    input_path=input_path,
                    output_path=root / "audit.json",
                )

    def test_run_audit_writes_report_and_aggregates_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "evaluation_results.json"
            output_path = root / "codex_audit_results.json"
            input_path.write_text(
                json.dumps({"per_question": [sample_record()]}),
                encoding="utf-8",
            )

            with (
                patch("codex_audit.resolve_codex_bin", return_value="codex"),
                patch("codex_audit.run_codex_audit", return_value=audit_payload()),
            ):
                report = codex_audit.run_audit(
                    input_path=input_path,
                    output_path=output_path,
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(1, report["metadata"]["question_count"])
            self.assertEqual(0.75, report["aggregate_scores"]["faithfulness"])
            self.assertEqual(
                1,
                report["disagreement_summary"]["ragas_disagreement_count"],
            )

    def test_run_audit_reuses_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "evaluation_results.json"
            output_path = root / "codex_audit_results.json"
            completed_record = {
                "question_number": 1,
                "question": "What failed?",
                **audit_payload(),
            }
            input_path.write_text(
                json.dumps({"per_question": [sample_record()]}),
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps({"per_question": [completed_record]}),
                encoding="utf-8",
            )

            with (
                patch("codex_audit.resolve_codex_bin", return_value="codex"),
                patch("codex_audit.run_codex_audit") as run_codex,
            ):
                report = codex_audit.run_audit(
                    input_path=input_path,
                    output_path=output_path,
                )

            run_codex.assert_not_called()
            self.assertEqual(
                "The answer is mostly grounded.",
                report["per_question"][0]["overall_summary"],
            )

    def test_aggregate_scores_ignores_errored_samples(self) -> None:
        good = {"scores": audit_payload()["scores"]}
        bad = {"audit_error": "Codex failed"}

        aggregate = codex_audit.aggregate_scores([good, bad])

        self.assertEqual(0.75, aggregate["answer_correctness"])


if __name__ == "__main__":
    unittest.main()
