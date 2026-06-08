import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


_ensure_nltk_resource("tokenizers/punkt", "punkt")
_ensure_nltk_resource("tokenizers/punkt_tab/english", "punkt_tab")
_ensure_nltk_resource("corpora/stopwords", "stopwords")


STOP_WORDS = set(stopwords.words("english"))

ACADEMIC_NOISE_WORDS = {
    "doi", "figure", "fig", "table", "tables", "et", "al", "available",
    "online", "accessed", "study", "studies", "used", "using", "use",
    "paper", "review", "introduction", "published", "received", "accepted",
    "copyright", "frontiers", "org", "med", "clinic",
}

REFERENCE_PATTERNS = [
    r"\bdoi\b",
    r"\bet al\.\b",
    r"\bavailable online at\b",
    r"\baccessed\b",
    r"http[s]?://",
    r"www\.",
    r"\bvol\.\b",
    r"\bpp\.\b",
]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Any
    lexical_score: float
    semantic_score: float
    hybrid_score: float


@dataclass(frozen=True)
class RankedChunk:
    chunk: Any
    lexical_score: float
    semantic_score: float
    hybrid_score: float
    rerank_score: float


@dataclass(frozen=True)
class TextSpan:
    text: str
    line_start: int
    line_end: int
    chunk_lines: list[str]
    parent_rerank_score: float


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_display_text(text: str) -> str:
    text = _normalize_whitespace(text)
    text = re.sub(r"^[•\-\u2022]+\s*", "", text)
    text = re.sub(r"\n[•\-\u2022]+\s*", "\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _looks_like_year_token(token: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}", token))


def _looks_like_reference_text(text: str) -> bool:
    text_lower = text.lower().strip()

    if any(re.search(pattern, text_lower) for pattern in REFERENCE_PATTERNS):
        return True

    # Many reference entries have author commas plus a year in parentheses.
    if re.search(r"\(\d{4}\)", text) and "," in text and len(text.split()) > 10:
        return True

    return False


def _looks_like_caption(text: str) -> bool:
    text_lower = text.strip().lower()

    if re.fullmatch(r"(table|figure|fig)\s*\d+[a-zA-Z]?\s*[:.\-]?\s*.*", text_lower):
        return True

    if text_lower.startswith(("table ", "figure ", "fig. ", "fig ")) and len(text.split()) <= 18:
        return True

    return False


def _looks_like_table_dense_text(text: str) -> bool:
    text_lower = text.lower()

    table_markers = [
        "model =",
        "references =",
        "data training set",
        "accuracy/specificity/sensitivity",
        "accuracy/specificity/ sensitivity",
    ]

    marker_hits = sum(text_lower.count(marker) for marker in table_markers)

    if marker_hits >= 3:
        return True

    if text_lower.startswith("table ") and len(text_lower.split()) > 30:
        return True

    if text_lower.count("auc") >= 3 and text_lower.count("references =") >= 1:
        return True

    return False


def _is_low_value_text(text: str, *, for_summary: bool = False) -> bool:
    cleaned = _clean_display_text(text)

    if not cleaned:
        return True

    if _looks_like_reference_text(cleaned):
        return True

    if _looks_like_caption(cleaned):
        return True

    if for_summary and _looks_like_table_dense_text(cleaned):
        return True

    if len(cleaned.split()) < 6:
        return True

    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    if alpha_chars < 20:
        return True

    return False


def clean_words(text: str, extra_stopwords: set[str] | None = None) -> list[str]:
    stopword_set = STOP_WORDS.copy()

    if extra_stopwords:
        stopword_set.update(extra_stopwords)

    words = word_tokenize(text.lower())
    cleaned_words = []

    for word in words:
        if not word.isalnum():
            continue
        if word in stopword_set:
            continue
        if len(word) <= 2:
            continue
        if _looks_like_year_token(word):
            continue

        cleaned_words.append(word)

    return cleaned_words


def extract_keywords(text: str, top_n: int = 12) -> list[tuple[str, int]]:
    cleaned_text = re.sub(r"http[s]?://\S+", " ", text)
    cleaned_text = re.sub(r"\bdoi[:\s]\S+", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bet al\.\b", " ", cleaned_text, flags=re.IGNORECASE)

    words = clean_words(cleaned_text, extra_stopwords=ACADEMIC_NOISE_WORDS)
    counts = Counter(words)
    return counts.most_common(top_n)


def _normalize_for_similarity(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def _token_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(clean_words(text_a))
    tokens_b = set(clean_words(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    return intersection / union if union else 0.0


def _is_duplicate_text(text_a: str, text_b: str) -> bool:
    normalized_a = _normalize_for_similarity(text_a)
    normalized_b = _normalize_for_similarity(text_b)

    sequence_ratio = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    jaccard_ratio = _token_jaccard_similarity(normalized_a, normalized_b)

    return sequence_ratio >= 0.82 or jaccard_ratio >= 0.78


def summarize_chunks(
    chunk_texts: list[str],
    chunk_embeddings,
    max_items: int = 5,
    diversity_lambda: float = 0.72,
) -> str:
    if not chunk_texts:
        return ""

    embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    filtered_indices = []
    filtered_texts = []

    for idx, text in enumerate(chunk_texts):
        cleaned = _clean_display_text(text)

        if _is_low_value_text(cleaned, for_summary=True):
            continue

        if any(_is_duplicate_text(cleaned, kept_text) for kept_text in filtered_texts):
            continue

        filtered_indices.append(idx)
        filtered_texts.append(cleaned)

    if not filtered_indices:
        fallback_texts = [
            _clean_display_text(text)
            for text in chunk_texts
            if not _is_low_value_text(text, for_summary=True)
        ]
        return "\n".join(f"- {text}" for text in fallback_texts[:max_items])

    candidate_embeddings = embeddings[filtered_indices]
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_embeddings = candidate_embeddings / np.clip(norms, 1e-12, None)

    # Central chunks are good summary candidates, but MMR keeps them from being too repetitive.
    centroid = candidate_embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    centrality_scores = candidate_embeddings @ centroid

    selected = []
    remaining = list(range(len(filtered_indices)))

    while remaining and len(selected) < max_items:
        if not selected:
            best_idx = max(remaining, key=lambda i: centrality_scores[i])
        else:
            def mmr_score(i: int) -> float:
                relevance = centrality_scores[i]
                diversity_penalty = max(
                    float(candidate_embeddings[i] @ candidate_embeddings[j])
                    for j in selected
                )
                return (diversity_lambda * relevance) - ((1 - diversity_lambda) * diversity_penalty)

            best_idx = max(remaining, key=mmr_score)

        selected.append(best_idx)
        remaining.remove(best_idx)

    selected.sort(key=lambda i: filtered_indices[i])

    summary_items = [filtered_texts[i] for i in selected]
    return "\n".join(f"- {item}" for item in summary_items)


def retrieve_relevant_chunks(
    query: str,
    chunks,
    top_k: int = 8,
    min_score: float = 0.05,
):
    if not chunks or not query.strip():
        return []

    chunk_texts = [chunk.text for chunk in chunks]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        chunk_vectors = vectorizer.fit_transform(chunk_texts)
        query_vector = vectorizer.transform([query])
    except ValueError:
        return []

    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    ranked_indices = similarities.argsort()[::-1]

    results = []

    for i in ranked_indices:
        score = float(similarities[i])

        if score >= min_score:
            results.append((chunks[i], score))

        if len(results) == top_k:
            break

    return results


def hybrid_retrieve_chunks(
    query: str,
    chunks,
    embedding_model,
    chunk_embeddings,
    top_k: int = 8,
    lexical_top_k: int = 12,
    semantic_top_k: int = 12,
    lexical_min_score: float = 0.02,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    if not chunks or not query.strip():
        return []

    lexical_results = retrieve_relevant_chunks(
        query,
        chunks,
        top_k=lexical_top_k,
        min_score=lexical_min_score,
    )

    lexical_rank = {}
    lexical_score_map = {}

    for rank_idx, (chunk, score) in enumerate(lexical_results, start=1):
        lexical_rank[chunk.index] = rank_idx
        lexical_score_map[chunk.index] = float(score)

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    if len(chunk_embeddings) != len(chunks):
        return []

    semantic_scores = np.dot(chunk_embeddings, query_embedding)
    semantic_indices = semantic_scores.argsort()[::-1][:semantic_top_k]

    semantic_rank = {}
    semantic_score_map = {}

    for rank_idx, chunk_idx in enumerate(semantic_indices, start=1):
        chunk_idx = int(chunk_idx)
        semantic_rank[chunk_idx] = rank_idx
        semantic_score_map[chunk_idx] = float(semantic_scores[chunk_idx])

    candidate_indices = set(lexical_rank.keys()).union(set(semantic_rank.keys()))
    fused_results = []

    for chunk_idx in candidate_indices:
        chunk = chunks[chunk_idx]
        hybrid_score = 0.0

        if chunk_idx in lexical_rank:
            hybrid_score += lexical_weight * (1.0 / (rrf_k + lexical_rank[chunk_idx]))

        if chunk_idx in semantic_rank:
            hybrid_score += semantic_weight * (1.0 / (rrf_k + semantic_rank[chunk_idx]))

        fused_results.append(
            RetrievedChunk(
                chunk=chunk,
                lexical_score=lexical_score_map.get(chunk_idx, 0.0),
                semantic_score=semantic_score_map.get(chunk_idx, 0.0),
                hybrid_score=hybrid_score,
            )
        )

    fused_results.sort(key=lambda item: item.hybrid_score, reverse=True)
    return fused_results[:top_k]


def rerank_chunks(query: str, retrieved_results, reranker, top_k: int = 5) -> list[RankedChunk]:
    if not retrieved_results:
        return []

    pairs = [(query, item.chunk.text) for item in retrieved_results]
    rerank_scores = reranker.predict(pairs)

    ranked = []

    for item, rerank_score in zip(retrieved_results, rerank_scores):
        ranked.append(
            RankedChunk(
                chunk=item.chunk,
                lexical_score=float(item.lexical_score),
                semantic_score=float(item.semantic_score),
                hybrid_score=float(item.hybrid_score),
                rerank_score=float(rerank_score),
            )
        )

    ranked.sort(key=lambda item: item.rerank_score, reverse=True)
    return ranked[:top_k]


def filter_and_deduplicate_supports(
    reranked_results,
    max_results: int = 3,
    min_rerank_score: float = 0.0,
) -> list[RankedChunk]:
    if not reranked_results:
        return []

    filtered = []

    for ranked in reranked_results:
        if ranked.rerank_score < min_rerank_score:
            continue

        if _is_low_value_text(ranked.chunk.text):
            continue

        if any(_is_duplicate_text(ranked.chunk.text, kept.chunk.text) for kept in filtered):
            continue

        filtered.append(ranked)

        if len(filtered) == max_results:
            break

    return filtered


def _split_chunk_into_spans(ranked_chunk: RankedChunk) -> list[TextSpan]:
    raw_lines = [
        line.strip()
        for line in ranked_chunk.chunk.text.replace("\r", "\n").split("\n")
        if line.strip()
    ]

    spans = []
    seen = set()

    def add_span(span_text: str, line_start: int, line_end: int) -> None:
        cleaned = _clean_display_text(span_text)
        normalized = _normalize_for_similarity(cleaned)

        if len(cleaned) < 25:
            return

        if _is_low_value_text(cleaned):
            return

        if normalized in seen:
            return

        seen.add(normalized)
        spans.append(
            TextSpan(
                text=cleaned,
                line_start=line_start,
                line_end=line_end,
                chunk_lines=raw_lines,
                parent_rerank_score=ranked_chunk.rerank_score,
            )
        )

    for line_idx, line in enumerate(raw_lines):
        add_span(line, line_idx, line_idx)

        for sentence in sent_tokenize(line):
            add_span(sentence, line_idx, line_idx)

    for line_idx in range(len(raw_lines) - 1):
        combined = raw_lines[line_idx] + "\n" + raw_lines[line_idx + 1]
        add_span(combined, line_idx, line_idx + 1)

    return spans


def _score_span(query: str, span: TextSpan, rerank_score: float) -> float:
    query_terms = set(clean_words(query))
    span_terms = set(clean_words(span.text))
    overlap = len(query_terms.intersection(span_terms))

    score = float(rerank_score)
    score += 0.08 * span.parent_rerank_score
    score += 0.05 * overlap

    # Very long spans often include extra context that makes direct answers less sharp.
    score -= 0.0007 * len(span.text)

    return score


def _best_span_for_query(query: str, ranked_chunks: list[RankedChunk], reranker, max_chunks: int = 3) -> TextSpan | None:
    candidate_spans = []

    for ranked_chunk in ranked_chunks[:max_chunks]:
        candidate_spans.extend(_split_chunk_into_spans(ranked_chunk))

    if not candidate_spans:
        return None

    pairs = [(query, span.text) for span in candidate_spans]
    span_scores = reranker.predict(pairs)

    best_span = None
    best_score = float("-inf")

    for span, span_score in zip(candidate_spans, span_scores):
        score = _score_span(query, span, float(span_score))

        if score > best_score:
            best_score = score
            best_span = span

    return best_span


def _complete_span(span: TextSpan, max_extra_lines: int = 2, max_chars: int = 450) -> str:
    selected_lines = span.chunk_lines[span.line_start:span.line_end + 1]
    text = "\n".join(selected_lines).strip()

    if text.endswith((".", "!", "?")) and len(text) >= 60:
        return _clean_display_text(text)

    next_line_idx = span.line_end + 1
    added_lines = 0

    while next_line_idx < len(span.chunk_lines) and added_lines < max_extra_lines:
        next_line = span.chunk_lines[next_line_idx].strip()

        if not next_line:
            break

        if _is_low_value_text(next_line):
            break

        new_text = "\n".join(selected_lines + [next_line]).strip()

        if len(new_text) > max_chars:
            break

        selected_lines.append(next_line)
        added_lines += 1
        next_line_idx += 1

        if next_line.endswith((".", "!", "?")) and len(new_text) >= 60:
            break

    return _clean_display_text("\n".join(selected_lines))


def _truncate_text(text: str, max_chars: int) -> str:
    text = _clean_display_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def extract_support_excerpt(
    query: str,
    ranked_chunk: RankedChunk,
    reranker,
    context_lines: int = 1,
    max_chars: int = 500,
) -> str:
    spans = _split_chunk_into_spans(ranked_chunk)

    if not spans:
        return _truncate_text(ranked_chunk.chunk.text, max_chars)

    pairs = [(query, span.text) for span in spans]
    span_scores = reranker.predict(pairs)

    best_span = max(
        zip(spans, span_scores),
        key=lambda item: _score_span(query, item[0], float(item[1])),
    )[0]

    start_line = max(0, best_span.line_start - context_lines)
    end_line = min(len(best_span.chunk_lines) - 1, best_span.line_end + context_lines)

    excerpt_lines = best_span.chunk_lines[start_line:end_line + 1]
    excerpt_text = _clean_display_text("\n".join(excerpt_lines))

    if start_line > 0:
        excerpt_text = "...\n" + excerpt_text

    if end_line < len(best_span.chunk_lines) - 1:
        excerpt_text = excerpt_text + "\n..."

    return _truncate_text(excerpt_text, max_chars)


def extract_direct_answer(query: str, reranked_results, reranker, max_chunks: int = 2) -> str:
    if not reranked_results:
        return ""

    usable_results = [
        ranked
        for ranked in reranked_results
        if not _is_low_value_text(ranked.chunk.text)
    ]

    answer_sources = usable_results if usable_results else reranked_results
    best_span = _best_span_for_query(query, answer_sources, reranker, max_chunks=max_chunks)

    if best_span is None:
        return _truncate_text(answer_sources[0].chunk.text, max_chars=450)

    return _complete_span(best_span)