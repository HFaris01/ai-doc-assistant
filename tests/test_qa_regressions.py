"""Offline regression tests; external libraries, models, and UI are test doubles.

Run from the repository root: python -m unittest discover -s tests -v
The real application modules are imported and exercised without downloads.
"""

import importlib.util
import re
import sys
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
ANSWER = "The warranty lasts exactly two years from the original purchase date."
BACKGROUND = "This sentence provides unrelated background material about the document. "


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


class FakeStreamlit(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.submitted = True
        self.query = "How long is the warranty?"
        for name in ("set_page_config", "subheader", "write", "caption", "warning", "error", "success"):
            setattr(self, name, Mock())
        self.file_uploader = Mock(return_value=None)

    def cache_resource(self, **kwargs):
        return lambda function: function

    cache_data = cache_resource

    def form(self, *args, **kwargs):
        return nullcontext()

    spinner = form
    expander = form

    def tabs(self, labels):
        return [nullcontext() for _ in labels]

    def text_input(self, *args, **kwargs):
        return self.query

    def form_submit_button(self, *args, **kwargs):
        return self.submitted


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("This offline test must not load external resources")


def load_application():
    st = FakeStreamlit()
    dependencies = {
        "streamlit": st,
        "sentence_transformers": module(
            "sentence_transformers", CrossEncoder=forbidden, SentenceTransformer=forbidden,
        ),
        "nltk": module(
            "nltk", data=SimpleNamespace(find=lambda path: True), download=forbidden,
        ),
        "nltk.corpus": module("nltk.corpus", stopwords=SimpleNamespace(words=lambda language: [])),
        "nltk.tokenize": module(
            "nltk.tokenize",
            sent_tokenize=lambda text: re.split(r"(?<=[.!?])\s+", text),
            word_tokenize=lambda text: re.findall(r"\w+", text),
        ),
        "numpy": module("numpy"),
        "sklearn": module("sklearn"),
        "sklearn.feature_extraction": module("sklearn.feature_extraction"),
        "sklearn.feature_extraction.text": module(
            "sklearn.feature_extraction.text", TfidfVectorizer=forbidden,
        ),
        "sklearn.metrics": module("sklearn.metrics"),
        "sklearn.metrics.pairwise": module("sklearn.metrics.pairwise", cosine_similarity=forbidden),
        "ollama": module("ollama", Client=forbidden),
        "utils.docling_utils": module(
            "utils.docling_utils", DoclingChunk=Chunk,
            build_docling_converter=forbidden, parse_pdf_with_docling=forbidden,
        ),
    }
    with patch.dict(sys.modules, dependencies):
        loaded = []
        for name, relative_path in (
            ("utils.nlp_utils", "utils/nlp_utils.py"),
            ("utils.ollama_utils", "utils/ollama_utils.py"),
            ("app", "app.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
            target = importlib.util.module_from_spec(spec)
            sys.modules[name] = target
            spec.loader.exec_module(target)
            loaded.append(target)
    nlp, _, app = loaded
    return st, nlp, app


class QARegressionTests(unittest.TestCase):
    def setUp(self):
        self.st, self.nlp, self.app = load_application()
        self.reranker = SimpleNamespace(predict=Mock(side_effect=lambda pairs: [
            10.0 if text == ANSWER else -10.0 for query, text in pairs
        ]))
        self.app.get_embedding_model = Mock(return_value=object())
        self.app.get_reranker = Mock(return_value=self.reranker)
        self.app.generate_grounded_answer_ollama = Mock(return_value=ANSWER)

    def ranked(self, text=ANSWER, score=5.0, index=0):
        return self.nlp.RankedChunk(Chunk(index, text), 0.5, 0.5, 0.03, score)

    def prepare_retrieval(self, results):
        self.app.hybrid_retrieve_chunks = Mock(return_value=results)
        self.app.rerank_chunks = Mock(return_value=results)
        self.app.extract_support_excerpt = Mock(wraps=self.nlp.extract_support_excerpt)

    def render(self, mode="Grounded LLM (Ollama)", debug=False, model="test-model"):
        self.app.render_question_tab([Chunk(0, ANSWER)], [[1.0]], debug, mode, model)

    def assert_abstained(self):
        result = self.st.session_state["qa_result"]
        self.assertEqual(result["answer"], self.app.INSUFFICIENT_CONTEXT_MESSAGE)
        self.assertEqual(result["support_sections"], [])
        self.app.generate_grounded_answer_ollama.assert_not_called()
        self.reranker.predict.assert_not_called()

    def test_rejected_scores_never_reach_either_answer_mode(self):
        for mode in ("Extractive", "Grounded LLM (Ollama)"):
            with self.subTest(mode=mode):
                self.prepare_retrieval([self.ranked(score=-8.0)])
                self.render(mode)
                self.assert_abstained()

    def test_rejected_text_never_reaches_generation(self):
        self.prepare_retrieval([self.ranked("Available online at https://example.org/reference.")])
        self.render()
        self.assert_abstained()

    def test_empty_retrieval_replaces_previous_answer_with_abstention(self):
        self.st.session_state["qa_result"] = {"answer": "Old answer"}
        self.prepare_retrieval([])
        self.render()
        self.assert_abstained()
        self.app.get_reranker.assert_not_called()

    def test_empty_reranking_abstains(self):
        self.prepare_retrieval([self.ranked()])
        self.app.rerank_chunks.return_value = []
        self.render()
        self.assert_abstained()

    def test_only_accepted_evidence_reaches_ollama_and_display(self):
        accepted = self.ranked(index=1)
        self.prepare_retrieval([self.ranked(BACKGROUND, score=-8.0), accepted])
        self.render()
        self.app.generate_grounded_answer_ollama.assert_called_once_with(
            query=self.st.query, context_chunks=[ANSWER], model_name="test-model",
        )
        self.assertEqual(self.st.session_state["qa_result"]["support_sections"], [(accepted, ANSWER)])

    def test_empty_sources_skip_generation(self):
        for mode in ("Extractive", "Grounded LLM (Ollama)"):
            answer = self.app.generate_answer(self.st.query, mode, "test-model", [], self.reranker)
            self.assertEqual(answer, self.app.INSUFFICIENT_CONTEXT_MESSAGE)
        self.app.generate_grounded_answer_ollama.assert_not_called()
        self.reranker.predict.assert_not_called()

    def test_extractive_helper_does_not_restore_rejected_sources(self):
        for ranked in (self.ranked(score=-8.0), self.ranked("https://example.org reference text")):
            self.assertEqual(self.nlp.extract_direct_answer(self.st.query, [ranked], self.reranker), "")
        self.reranker.predict.assert_not_called()

    def test_extractive_answer_preserves_selected_sentence(self):
        paragraph = BACKGROUND * 12 + ANSWER
        answer = self.nlp.extract_direct_answer(self.st.query, [self.ranked(paragraph)], self.reranker)
        self.assertEqual(answer, ANSWER)

    def test_excerpts_keep_selected_sentence_at_start_middle_and_end(self):
        for paragraph in (ANSWER + " " + BACKGROUND * 12, BACKGROUND * 6 + ANSWER + " " + BACKGROUND * 6, BACKGROUND * 12 + ANSWER):
            with self.subTest(position=paragraph.index(ANSWER)):
                excerpt = self.nlp.extract_support_excerpt(self.st.query, self.ranked(paragraph), self.reranker)
                self.assertIn(ANSWER, excerpt)
                self.assertLessEqual(len(excerpt), 500)

    def test_excerpt_prioritizes_evidence_over_long_neighboring_lines(self):
        text = BACKGROUND * 12 + "\n" + ANSWER + "\n" + BACKGROUND * 12
        excerpt = self.nlp.extract_support_excerpt(self.st.query, self.ranked(text), self.reranker)
        self.assertIn(ANSWER, excerpt)
        self.assertLessEqual(len(excerpt), 500)

    def test_span_offsets_match_normalized_source(self):
        text = "  - " + BACKGROUND + ANSWER + "\n\n" + BACKGROUND
        for span in self.nlp._split_chunk_into_spans(self.ranked(text)):
            source = "\n".join(span.chunk_lines)
            self.assertEqual(source[span.char_start:span.char_end], span.text)

    def test_negative_values_survive_cleanup_answers_and_excerpts(self):
        for value in ("-12", "-12.5", "-.5", "\u221212"):
            sentence = value + " degrees Celsius was the lowest temperature measured during the winter survey."
            for bullet in ("", "- ", "\u2022 ", "- - ", "\u2022- "):
                with self.subTest(value=value, bullet=bullet):
                    ranked = self.ranked(bullet + sentence)
                    cleaned = self.nlp._clean_display_text(ranked.chunk.text)
                    self.assertEqual(cleaned, sentence)
                    self.assertEqual(self.nlp._clean_display_text(cleaned), sentence)
                    self.assertEqual(
                        self.nlp.extract_direct_answer("What was the lowest temperature?", [ranked], self.reranker),
                        sentence,
                    )
                    self.assertEqual(
                        self.nlp.extract_support_excerpt("What was the lowest temperature?", ranked, self.reranker),
                        sentence,
                    )
                    for span in self.nlp._split_chunk_into_spans(ranked):
                        source = "\n".join(span.chunk_lines)
                        self.assertEqual(source[span.char_start:span.char_end], span.text)

    def test_multiline_bullet_cleanup_preserves_negative_values(self):
        text = "- -12 degrees Celsius\n  \u2022 -15 percent\n- -.5 degrees Celsius"
        expected = "-12 degrees Celsius\n-15 percent\n-.5 degrees Celsius"
        cleaned = self.nlp._clean_display_text(text)
        self.assertEqual(cleaned, expected)
        self.assertEqual(self.nlp._clean_display_text(cleaned), expected)

    def test_excerpt_preserves_full_text_up_to_the_limit(self):
        for length in range(491, 501):
            prefix = "The warranty applies to all eligible products "
            sentence = prefix + "x" * (length - len(prefix) - len(" only.")) + " only."
            with self.subTest(length=length):
                excerpt = self.nlp.extract_support_excerpt(self.st.query, self.ranked(sentence), self.reranker)
                self.assertEqual(excerpt, sentence)

    def test_excerpt_reserves_only_the_markers_it_displays(self):
        for before, after in ((False, True), (True, False), (True, True)):
            prefix = "The warranty applies to all eligible products "
            sentence_length = 500 - 4 * (before + after)
            sentence = prefix + "x" * (sentence_length - len(prefix) - len(" only.")) + " only."
            reranker = SimpleNamespace(predict=lambda pairs: [
                10.0 if text == sentence else -10.0 for query, text in pairs
            ])
            for separator in (" ", "\n"):
                for context_lines in (0, 1):
                    with self.subTest(before=before, after=after, separator=separator, context_lines=context_lines):
                        text = separator.join(
                            ([BACKGROUND.strip()] if before else []) + [sentence]
                            + ([BACKGROUND.strip()] if after else [])
                        )
                        excerpt = self.nlp.extract_support_excerpt(
                            self.st.query, self.ranked(text), reranker, context_lines=context_lines,
                        )
                        expected = ("...\n" if before else "") + sentence + ("\n..." if after else "")
                        self.assertEqual(excerpt, expected)

    def test_complete_short_sentence_does_not_append_next_line(self):
        for text in (
            "All submitted applications require a valid signature.",
            'The policy states: "All applications require a valid signature."',
        ):
            with self.subTest(text=text):
                spans = self.nlp._split_chunk_into_spans(self.ranked(text + "\n" + BACKGROUND))
                span = next(span for span in spans if span.text == text)
                self.assertEqual(self.nlp._complete_span(span), text)

    def test_truncation_keeps_complete_words_at_the_boundary(self):
        self.assertEqual(self.nlp._truncate_text("alpha beta gamma", 14), "alpha beta...")

    def test_incomplete_span_can_continue_on_next_line(self):
        first = "The original warranty remains valid for"
        second = "exactly two years from the original purchase date."
        spans = self.nlp._split_chunk_into_spans(self.ranked(first + "\n" + second))
        span = next(span for span in spans if span.text == first)
        self.assertEqual(self.nlp._complete_span(span), first + "\n" + second)

    def test_long_answers_and_excerpts_respect_total_character_limits(self):
        text = "The warranty applies to " + "all eligible products " * 60 + "only."
        ranked = self.ranked(text)
        answer = self.nlp.extract_direct_answer(self.st.query, [ranked], self.reranker)
        self.assertLessEqual(len(answer), 450)
        self.assertTrue(answer.endswith("..."))
        for limit in (0, 1, 3, 8, 20, 500):
            with self.subTest(limit=limit):
                excerpt = self.nlp.extract_support_excerpt(self.st.query, ranked, self.reranker, max_chars=limit)
                self.assertLessEqual(len(excerpt), limit)

    def test_rerun_keeps_answer_evidence_and_original_mode_without_recomputing(self):
        self.prepare_retrieval([self.ranked()])
        self.render()
        original = self.st.session_state["qa_result"]
        self.st.submitted = False
        self.st.write.reset_mock()
        self.st.caption.reset_mock()
        self.render(mode="Extractive", debug=True, model="different-model")
        self.assertIs(self.st.session_state["qa_result"], original)
        self.st.write.assert_any_call(ANSWER)
        self.st.caption.assert_any_call("Mode: Grounded LLM (Ollama) | Model: test-model")
        self.assertTrue(any("rerank:" in call.args[0] for call in self.st.caption.call_args_list))
        self.app.hybrid_retrieve_chunks.assert_called_once()
        self.app.rerank_chunks.assert_called_once()
        self.app.generate_grounded_answer_ollama.assert_called_once()
        self.app.extract_support_excerpt.assert_called_once()
        self.app.get_reranker.assert_called_once()

    def test_blank_submission_keeps_labeled_previous_answer(self):
        self.prepare_retrieval([self.ranked()])
        self.render()
        original = self.st.session_state["qa_result"]
        self.st.query = "   "
        self.render()
        self.assertIs(self.st.session_state["qa_result"], original)
        self.st.warning.assert_called_with("Please enter a question.")
        self.app.hybrid_retrieve_chunks.assert_called_once()

    def test_document_replacement_or_removal_clears_answer_and_question(self):
        self.app.apply_page_style = Mock()
        self.app.render_sidebar_tools = Mock(return_value=(False, "Extractive", "test-model"))
        self.app.render_header = Mock()
        for name in ("render_summary_tab", "render_keywords_tab", "render_question_tab", "render_raw_text_tab"):
            setattr(self.app, name, Mock())
        self.app.get_cached_docling_parse = Mock(return_value=(ANSWER, [ANSWER]))
        self.app.get_cached_chunk_embeddings = Mock(return_value=[[1.0]])
        for replacement in (b"different document", None):
            with self.subTest(replacement=replacement):
                self.st.session_state.update(
                    qa_document_id=sha256(b"original").hexdigest(), qa_result={"answer": ANSWER}, qa_query="Old question",
                )
                self.st.file_uploader.return_value = SimpleNamespace(name="same.pdf", getvalue=lambda: b"original")
                self.app.main()
                self.assertIn("qa_result", self.st.session_state)
                self.st.file_uploader.return_value = (
                    SimpleNamespace(name="same.pdf", getvalue=lambda: replacement) if replacement is not None else None
                )
                self.app.main()
                self.assertNotIn("qa_result", self.st.session_state)
                self.assertEqual(self.st.session_state["qa_query"], "")


if __name__ == "__main__":
    unittest.main()
