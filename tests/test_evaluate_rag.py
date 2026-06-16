import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

import evaluate_rag
from app import (
    HybridSearchEngine,
    ResourceExhausted,
    RetrievedChunk,
    evaluation_process_error,
)


class OllamaPreflightTests(unittest.TestCase):
    @patch("evaluate_rag.urllib.request.urlopen")
    def test_check_ollama_accepts_installed_model(self, urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        urlopen.return_value = response

        with patch(
            "evaluate_rag.json.load",
            return_value={"models": [{"name": "qwen2.5:3b"}]},
        ):
            ready, error = evaluate_rag.check_ollama(
                base_url="http://127.0.0.1:11434/",
                model_name="qwen2.5:3b",
            )

        self.assertTrue(ready)
        self.assertEqual("", error)
        urlopen.assert_called_once_with(
            "http://127.0.0.1:11434/api/tags",
            timeout=3.0,
        )

    @patch("evaluate_rag.urllib.request.urlopen")
    def test_check_ollama_explains_missing_model(self, urlopen) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        urlopen.return_value = response

        with patch(
            "evaluate_rag.json.load",
            return_value={"models": [{"name": "llama3.2:3b"}]},
        ):
            ready, error = evaluate_rag.check_ollama(
                base_url="http://127.0.0.1:11434",
                model_name="qwen2.5:3b",
            )

        self.assertFalse(ready)
        self.assertIn("ollama pull qwen2.5:3b", error)


class EvaluationRoutingTests(unittest.TestCase):
    def successful_ragas_result(self, row_count: int) -> MagicMock:
        rows = [
            {metric_name: 0.8 for metric_name in evaluate_rag.METRIC_NAMES}
            for _ in range(row_count)
        ]
        evaluation_result = MagicMock()
        evaluation_result.to_pandas.return_value = pd.DataFrame(rows)
        return evaluation_result

    def test_ragas_uses_local_judge_and_one_answer_call_per_question(self) -> None:
        questions = [
            {
                "question": "Question one?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point one"],
            },
            {
                "question": "Question two?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point two"],
            },
        ]
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=1,
            page_number=1,
            text="Relevant context.",
        )
        engine = MagicMock()
        engine.answer.side_effect = [
            ({"format": "legacy", "content": "Answer one."}, [chunk]),
            ({"format": "legacy", "content": "Answer two."}, [chunk]),
        ]
        engine.vector_retriever._embed_model = object()
        engine.llm = object()

        rows = [
            {
                metric_name: 0.8
                for metric_name in evaluate_rag.METRIC_NAMES
            }
            for _ in questions
        ]
        evaluation_result = MagicMock()
        evaluation_result.to_pandas.return_value = pd.DataFrame(rows)
        local_judge = object()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.json"
            output_path = root / "results.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            with (
                patch(
                    "evaluate_rag.create_ragas_judge",
                    return_value=local_judge,
                ) as create_judge,
                patch(
                    "evaluate_rag.LlamaIndexEmbeddingsWrapper",
                    return_value=object(),
                ),
                patch(
                    "evaluate_rag.evaluate",
                    return_value=evaluation_result,
                ) as ragas_evaluate,
            ):
                report = evaluate_rag.run_evaluation(
                    api_key="unused-by-test",
                    model_name="models/gemini-2.5-flash",
                    questions_path=questions_path,
                    output_path=output_path,
                    engine=engine,
                    evaluation_provider="ollama",
                    ollama_model="qwen2.5:3b",
                    ollama_base_url="http://127.0.0.1:11434",
                )

        self.assertEqual(2, engine.answer.call_count)
        for call in engine.answer.call_args_list:
            self.assertFalse(call.kwargs["allow_legacy_fallback"])
        create_judge.assert_called_once()
        self.assertIs(local_judge, ragas_evaluate.call_args.kwargs["llm"])
        self.assertIsNot(engine.llm, ragas_evaluate.call_args.kwargs["llm"])
        self.assertEqual("ollama", report["metadata"]["evaluation_llm_provider"])
        self.assertTrue(
            all(
                score is not None
                for item in report["per_question"]
                for score in item["scores"].values()
            )
        )

    def test_quota_error_checkpoints_completed_answers(self) -> None:
        questions = [
            {
                "question": "Question one?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point one"],
            },
            {
                "question": "Question two?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point two"],
            },
        ]
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=1,
            page_number=1,
            text="Relevant context.",
        )
        engine = MagicMock()
        engine.answer.side_effect = [
            ({"format": "legacy", "content": "Answer one."}, [chunk]),
            ResourceExhausted("quota"),
        ]
        engine.vector_retriever._embed_model = object()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.json"
            output_path = root / "results.json"
            answer_cache_path = root / "answer_cache.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")

            with self.assertRaises(ResourceExhausted):
                evaluate_rag.run_evaluation(
                    api_key="unused-by-test",
                    model_name="models/gemini-2.5-flash",
                    questions_path=questions_path,
                    output_path=output_path,
                    engine=engine,
                    evaluation_provider="ollama",
                    ollama_model="qwen2.5:3b",
                    ollama_base_url="http://127.0.0.1:11434",
                    answer_cache_path=answer_cache_path,
                )

            cache = json.loads(answer_cache_path.read_text(encoding="utf-8"))
            self.assertEqual(evaluate_rag.ANSWER_CACHE_SCHEMA_VERSION, cache["schema_version"])
            self.assertEqual(1, len(cache["records"]))
            self.assertEqual("Question one?", cache["records"][0]["question"])
            self.assertEqual("Answer one.", cache["records"][0]["answer"])
            self.assertFalse(output_path.exists())

    def test_resume_reuses_cached_answers_and_generates_missing_ones(self) -> None:
        questions = [
            {
                "question": "Question one?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point one"],
            },
            {
                "question": "Question two?",
                "expected_sources": ["source.pdf"],
                "answer_points": ["Point two"],
            },
        ]
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=2,
            page_number=1,
            text="Fresh context.",
        )
        engine = MagicMock()
        engine.answer.return_value = (
            {"format": "legacy", "content": "Answer two."},
            [chunk],
        )
        engine.vector_retriever._embed_model = object()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.json"
            output_path = root / "results.json"
            answer_cache_path = root / "answer_cache.json"
            questions_path.write_text(json.dumps(questions), encoding="utf-8")
            evaluate_rag.write_answer_cache(
                answer_cache_path,
                [
                    {
                        **evaluate_rag.answer_cache_signature(
                            model_name="models/gemini-2.5-flash",
                            question_number=1,
                            test_case=questions[0],
                        ),
                        "answer": "Cached answer one.",
                        "contexts": ["Cached context."],
                        "ground_truth": "Point one.",
                        "retrieved_sources": ["source.pdf"],
                        "retrieval_diagnostics": {
                            "expected_source_coverage": 1.0,
                            "missing_expected_sources": [],
                        },
                    }
                ],
            )

            with (
                patch("evaluate_rag.create_ragas_judge", return_value=object()),
                patch("evaluate_rag.LlamaIndexEmbeddingsWrapper", return_value=object()),
                patch(
                    "evaluate_rag.evaluate",
                    return_value=self.successful_ragas_result(2),
                ) as ragas_evaluate,
            ):
                report = evaluate_rag.run_evaluation(
                    api_key="unused-by-test",
                    model_name="models/gemini-2.5-flash",
                    questions_path=questions_path,
                    output_path=output_path,
                    engine=engine,
                    evaluation_provider="ollama",
                    ollama_model="qwen2.5:3b",
                    ollama_base_url="http://127.0.0.1:11434",
                    answer_cache_path=answer_cache_path,
                )

        engine.answer.assert_called_once_with(
            "Question two?",
            allow_legacy_fallback=False,
        )
        dataset = ragas_evaluate.call_args.args[0]
        self.assertEqual(2, len(dataset.samples))
        self.assertEqual("Cached answer one.", report["per_question"][0]["answer"])
        self.assertEqual("Answer two.", report["per_question"][1]["answer"])

    def test_stale_cached_answers_are_ignored(self) -> None:
        question = {
            "question": "Question?",
            "expected_sources": ["source.pdf"],
            "answer_points": ["Point"],
        }
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=1,
            page_number=1,
            text="Fresh context.",
        )
        engine = MagicMock()
        engine.answer.return_value = (
            {"format": "legacy", "content": "Fresh answer."},
            [chunk],
        )
        engine.vector_retriever._embed_model = object()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.json"
            output_path = root / "results.json"
            answer_cache_path = root / "answer_cache.json"
            questions_path.write_text(json.dumps([question]), encoding="utf-8")
            evaluate_rag.write_answer_cache(
                answer_cache_path,
                [
                    {
                        **evaluate_rag.answer_cache_signature(
                            model_name="models/gemini-1.5-flash",
                            question_number=1,
                            test_case=question,
                        ),
                        "answer": "Stale answer.",
                        "contexts": ["Stale context."],
                        "ground_truth": "Point.",
                        "retrieved_sources": ["source.pdf"],
                        "retrieval_diagnostics": {
                            "expected_source_coverage": 1.0,
                            "missing_expected_sources": [],
                        },
                    }
                ],
            )

            with (
                patch("evaluate_rag.create_ragas_judge", return_value=object()),
                patch("evaluate_rag.LlamaIndexEmbeddingsWrapper", return_value=object()),
                patch(
                    "evaluate_rag.evaluate",
                    return_value=self.successful_ragas_result(1),
                ),
            ):
                report = evaluate_rag.run_evaluation(
                    api_key="unused-by-test",
                    model_name="models/gemini-2.5-flash",
                    questions_path=questions_path,
                    output_path=output_path,
                    engine=engine,
                    evaluation_provider="ollama",
                    ollama_model="qwen2.5:3b",
                    ollama_base_url="http://127.0.0.1:11434",
                    answer_cache_path=answer_cache_path,
                )

        engine.answer.assert_called_once_with("Question?", allow_legacy_fallback=False)
        self.assertEqual("Fresh answer.", report["per_question"][0]["answer"])

    def test_quota_error_does_not_trigger_legacy_fallback(self) -> None:
        engine = HybridSearchEngine.__new__(HybridSearchEngine)
        engine.llm = MagicMock()
        engine.llm.structured_predict.side_effect = ResourceExhausted("quota")
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=1,
            page_number=1,
            text="Context.",
        )
        engine.retrieve = MagicMock(return_value=[chunk])

        with self.assertRaises(ResourceExhausted):
            engine.answer("Question?")

        engine.llm.complete.assert_not_called()

    def test_invalid_scores_do_not_overwrite_previous_report(self) -> None:
        question = {
            "question": "Question?",
            "expected_sources": ["source.pdf"],
            "answer_points": ["Point"],
        }
        chunk = RetrievedChunk(
            node_id="node",
            source_file="source.pdf",
            chunk_id=1,
            page_number=1,
            text="Relevant context.",
        )
        engine = MagicMock()
        engine.answer.return_value = (
            {"format": "legacy", "content": "Answer."},
            [chunk],
        )
        engine.vector_retriever._embed_model = object()

        evaluation_result = MagicMock()
        evaluation_result.to_pandas.return_value = pd.DataFrame(
            [{metric_name: None for metric_name in evaluate_rag.METRIC_NAMES}]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            questions_path = root / "questions.json"
            output_path = root / "results.json"
            questions_path.write_text(json.dumps([question]), encoding="utf-8")
            output_path.write_text('{"previous": true}\n', encoding="utf-8")

            with (
                patch(
                    "evaluate_rag.create_ragas_judge",
                    return_value=object(),
                ),
                patch(
                    "evaluate_rag.LlamaIndexEmbeddingsWrapper",
                    return_value=object(),
                ),
                patch(
                    "evaluate_rag.evaluate",
                    return_value=evaluation_result,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    evaluate_rag.run_evaluation(
                        api_key="unused-by-test",
                        model_name="models/gemini-2.5-flash",
                        questions_path=questions_path,
                        output_path=output_path,
                        engine=engine,
                        evaluation_provider="ollama",
                        ollama_model="qwen2.5:3b",
                        ollama_base_url="http://127.0.0.1:11434",
                    )

            self.assertEqual(
                '{"previous": true}\n',
                output_path.read_text(encoding="utf-8"),
            )

    def test_evaluator_failures_have_distinct_messages(self) -> None:
        quota = evaluation_process_error(1, "RESOURCE_EXHAUSTED Quota exceeded")
        unavailable = evaluation_process_error(1, "Could not reach Ollama")
        timeout = evaluation_process_error(1, "TimeoutError")
        malformed = evaluation_process_error(1, "Invalid JSON response")

        self.assertIn("Gemini API quota", quota)
        self.assertIn("reuse", quota)
        self.assertIn("Ollama is unavailable", unavailable)
        self.assertIn("timed out", timeout)
        self.assertIn("malformed structured output", malformed)


if __name__ == "__main__":
    unittest.main()
