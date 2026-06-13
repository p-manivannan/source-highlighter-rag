from __future__ import annotations

import unittest

from app import (
    AnswerClaim,
    EvidenceReference,
    GroundedAnswer,
    RetrievedChunk,
    is_complete_source_passage,
    validate_structured_answer,
)


SOURCE_TEXT = (
    "Czechia serves as the protecting power for U.S. interests in Syria in "
    "the absence of a U.S. mission. In May 2026, President Trump named U.S. "
    "Ambassador to Turkey Tom Barrack as Special Presidential Envoy to Syria "
    "and Iraq. The State Department's Syria Regional Platform includes "
    "personnel in Turkey and Jordan. Congress received a related funding request."
)
TOM_BARRACK_QUOTE = (
    "In May 2026, President Trump named U.S. Ambassador to Turkey Tom Barrack "
    "as Special Presidential Envoy to Syria and Iraq."
)


def source_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        node_id="node-7",
        source_file="IF11930.11.pdf",
        chunk_id=7,
        page_number=2,
        text=SOURCE_TEXT,
    )


class ClaimHighlightingTests(unittest.TestCase):
    def test_exact_sentence_becomes_exact_quote_highlight(self) -> None:
        answer = GroundedAnswer(
            status="answered",
            claims=[
                AnswerClaim(
                    text="Tom Barrack was designated as the presidential envoy.",
                    evidence=[
                        EvidenceReference(
                            source_file="IF11930.11.pdf",
                            chunk_id=7,
                            evidence_quote=TOM_BARRACK_QUOTE,
                        )
                    ],
                )
            ],
        )

        payload = validate_structured_answer(answer, [source_chunk()])

        occurrence = payload["citation_occurrences"][0]
        self.assertEqual("exact_quote", occurrence["highlight_mode"])
        self.assertEqual(TOM_BARRACK_QUOTE, occurrence["highlight_text"])

    def test_two_claims_in_same_chunk_get_distinct_occurrences(self) -> None:
        first_quote = (
            "Czechia serves as the protecting power for U.S. interests in Syria "
            "in the absence of a U.S. mission."
        )
        answer = GroundedAnswer(
            status="answered",
            claims=[
                AnswerClaim(
                    text="Czechia represents U.S. interests in Syria.",
                    evidence=[
                        EvidenceReference(
                            source_file="IF11930.11.pdf",
                            chunk_id=7,
                            evidence_quote=first_quote,
                        )
                    ],
                ),
                AnswerClaim(
                    text="Tom Barrack was named special envoy.",
                    evidence=[
                        EvidenceReference(
                            source_file="IF11930.11.pdf",
                            chunk_id=7,
                            evidence_quote=TOM_BARRACK_QUOTE,
                        )
                    ],
                ),
            ],
        )

        payload = validate_structured_answer(answer, [source_chunk()])
        occurrences = payload["citation_occurrences"]

        self.assertEqual([1, 2], [item["citation_number"] for item in occurrences])
        self.assertNotEqual(
            occurrences[0]["citation_id"],
            occurrences[1]["citation_id"],
        )
        self.assertNotEqual(
            occurrences[0]["highlight_text"],
            occurrences[1]["highlight_text"],
        )

    def test_two_or_three_contiguous_sentences_are_valid(self) -> None:
        two_sentences = (
            f"{TOM_BARRACK_QUOTE} The State Department's Syria Regional "
            "Platform includes personnel in Turkey and Jordan."
        )
        three_sentences = (
            "Czechia serves as the protecting power for U.S. interests in Syria "
            f"in the absence of a U.S. mission. {two_sentences}"
        )

        self.assertTrue(is_complete_source_passage(SOURCE_TEXT, two_sentences))
        self.assertTrue(is_complete_source_passage(SOURCE_TEXT, three_sentences))

    def test_invalid_or_overlong_quote_falls_back_to_chunk(self) -> None:
        four_sentences = SOURCE_TEXT
        answer = GroundedAnswer(
            status="answered",
            claims=[
                AnswerClaim(
                    text="A broad claim.",
                    evidence=[
                        EvidenceReference(
                            source_file="IF11930.11.pdf",
                            chunk_id=7,
                            evidence_quote=four_sentences,
                        )
                    ],
                )
            ],
        )

        payload = validate_structured_answer(answer, [source_chunk()])
        occurrence = payload["citation_occurrences"][0]

        self.assertEqual("chunk_fallback", occurrence["highlight_mode"])
        self.assertEqual(SOURCE_TEXT, occurrence["highlight_text"])

    def test_unretrieved_citation_is_not_exposed(self) -> None:
        answer = GroundedAnswer(
            status="answered",
            claims=[
                AnswerClaim(
                    text="An invented claim.",
                    evidence=[
                        EvidenceReference(
                            source_file="invented.pdf",
                            chunk_id=999,
                            evidence_quote="Invented evidence.",
                        )
                    ],
                )
            ],
        )

        with self.assertRaises(ValueError):
            validate_structured_answer(answer, [source_chunk()])

    def test_not_found_has_no_citations(self) -> None:
        payload = validate_structured_answer(
            GroundedAnswer(
                status="not_found",
                not_found_message="The documents do not answer this question.",
            ),
            [source_chunk()],
        )

        self.assertEqual([], payload["claims"])
        self.assertEqual([], payload["citation_occurrences"])


if __name__ == "__main__":
    unittest.main()
