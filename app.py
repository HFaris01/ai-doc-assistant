import os
from hashlib import sha256

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from utils.docling_utils import build_docling_converter, DoclingChunk, parse_pdf_with_docling
from utils.nlp_utils import (
    extract_direct_answer,
    extract_keywords,
    extract_support_excerpt,
    filter_and_deduplicate_supports,
    hybrid_retrieve_chunks,
    rerank_chunks,
    summarize_chunks,
)
from utils.ollama_utils import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    check_ollama_connection,
    generate_grounded_answer_ollama,
)


APP_TITLE = "AI Doc Assistant"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "gemma3:latest"

ANSWER_MODES = ["Extractive", "Grounded LLM (Ollama)"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


def apply_page_style() -> None:
    """Small UI tweaks to make the app easier to read in Streamlit."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 7rem;
                max-width: 1100px;
            }

            div[data-testid="stSidebar"] .stRadio label,
            div[data-testid="stSidebar"] .stTextInput label,
            div[data-testid="stSidebar"] .stToggle label {
                font-size: 0.95rem;
            }

            div[data-testid="stExpander"] summary {
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_docling_converter():
    # The converter is cached because building Docling resources can be slow.
    artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH")
    return build_docling_converter(artifacts_path=artifacts_path)


@st.cache_resource(show_spinner="Loading semantic retrieval model...")
def get_embedding_model():
    # This model creates vector embeddings for semantic search.
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading reranker model...")
def get_reranker():
    # The reranker compares the user question with retrieved chunks more carefully.
    return CrossEncoder(DEFAULT_RERANKER_MODEL)


@st.cache_data(show_spinner="Parsing and caching the PDF with Docling...")
def get_cached_docling_parse(file_bytes: bytes, filename: str) -> tuple[str, list[str]]:
    converter = get_docling_converter()

    full_text, chunks = parse_pdf_with_docling(
        file_bytes,
        filename=filename,
        converter=converter,
    )

    # Store only plain text in the cache. It is simpler and avoids caching custom objects.
    chunk_texts = [chunk.text for chunk in chunks]
    return full_text, chunk_texts


@st.cache_data(show_spinner="Encoding chunk embeddings...")
def get_cached_chunk_embeddings(chunk_texts: tuple[str, ...]) -> list[list[float]]:
    embedding_model = get_embedding_model()

    embeddings = embedding_model.encode(
        list(chunk_texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Streamlit's data cache is happier with simple Python lists than NumPy arrays.
    return embeddings.tolist()


def rebuild_chunks(chunk_texts: list[str]) -> list[DoclingChunk]:
    """Rebuild chunk objects after loading plain text from the cache."""
    return [
        DoclingChunk(index=i, text=text)
        for i, text in enumerate(chunk_texts)
    ]


def render_sidebar_tools(show_debug_default: bool = False) -> tuple[bool, str, str]:
    st.sidebar.header("Options")

    answer_mode = st.sidebar.radio(
        "Answer mode",
        ANSWER_MODES,
        index=1,
    )

    ollama_model_name = st.sidebar.text_input(
        "Ollama model name",
        value=DEFAULT_OLLAMA_MODEL,
        help="Use the exact model name returned by 'ollama list'.",
    )

    show_debug = st.sidebar.toggle(
        "Show retrieval diagnostics",
        value=show_debug_default,
    )

    if answer_mode == "Grounded LLM (Ollama)":
        ok, message = check_ollama_connection(ollama_model_name)

        if ok:
            st.sidebar.success(message)
        else:
            st.sidebar.warning(message)

    with st.sidebar.expander("Developer Tools"):
        st.caption("Use these only if the app needs a manual refresh.")

        if st.button("Clear parse cache", use_container_width=True):
            get_cached_docling_parse.clear()
            st.sidebar.success("Parse cache cleared.")

        if st.button("Clear embedding cache", use_container_width=True):
            get_cached_chunk_embeddings.clear()
            st.sidebar.success("Embedding cache cleared.")

        if st.button("Clear converter cache", use_container_width=True):
            get_docling_converter.clear()
            st.sidebar.success("Converter cache cleared.")

        if st.button("Clear model cache", use_container_width=True):
            get_embedding_model.clear()
            get_reranker.clear()
            st.sidebar.success("Model cache cleared.")

    return show_debug, answer_mode, ollama_model_name


def render_header() -> None:
    st.title(APP_TITLE)
    st.write(
        "Upload a PDF to ask questions about its contents."
    )


def render_summary_tab(chunk_texts: list[str], chunk_embeddings: list[list[float]]) -> None:
    st.subheader("Summary")

    summary = summarize_chunks(
        chunk_texts,
        chunk_embeddings,
        max_items=4,
    )

    st.markdown(summary if summary else "_No summary could be generated._")


def render_keywords_tab(full_text: str) -> None:
    st.subheader("Keywords")

    keywords = extract_keywords(full_text, top_n=15)

    if not keywords:
        st.info("No keywords could be extracted.")
        return

    for word, count in keywords:
        st.write(f"**{word}**: {count}")


def generate_answer(
    query: str,
    answer_mode: str,
    ollama_model_name: str,
    answer_source_results,
    reranker,
) -> str:
    # Only accepted evidence may reach either answer mode.
    if not answer_source_results:
        return INSUFFICIENT_CONTEXT_MESSAGE

    if answer_mode == "Grounded LLM (Ollama)":
        grounded_context_chunks = [
            item.chunk.text
            for item in answer_source_results[:3]
        ]

        try:
            return generate_grounded_answer_ollama(
                query=query,
                context_chunks=grounded_context_chunks,
                model_name=ollama_model_name,
            )
        except Exception as e:
            return f"Ollama answer generation failed: {e}"

    return extract_direct_answer(
        query,
        answer_source_results,
        reranker,
        max_chunks=2,
    ) or INSUFFICIENT_CONTEXT_MESSAGE


def render_support_sections(support_sections, show_debug: bool) -> None:
    if not support_sections:
        return

    st.write("### Retrieved passages")

    for i, (ranked, support_excerpt) in enumerate(support_sections, start=1):
        with st.expander(f"Passage {i}", expanded=(i == 1)):
            st.write(support_excerpt)

            if show_debug:
                st.caption(
                    f"lexical: {ranked.lexical_score:.3f} | "
                    f"semantic: {ranked.semantic_score:.3f} | "
                    f"hybrid: {ranked.hybrid_score:.4f} | "
                    f"rerank: {ranked.rerank_score:.3f}"
                )


def reset_question_state(document_id: str | None) -> None:
    """Clear the saved result when the document changes."""
    if st.session_state.get("qa_document_id") != document_id:
        st.session_state["qa_document_id"] = document_id
        st.session_state.pop("qa_result", None)
        # Deleting a widget key alone can restore its old frontend value.
        st.session_state["qa_query"] = ""


def render_question_tab(
    chunks: list[DoclingChunk],
    chunk_embeddings: list[list[float]],
    show_debug: bool,
    answer_mode: str,
    ollama_model_name: str,
) -> None:
    st.subheader("Ask a question")

    with st.form("qa_form", clear_on_submit=False):
        query = st.text_input(
            "Type a question about the document",
            key="qa_query",
            placeholder="Enter your question.",
        )
        ask_button = st.form_submit_button("Ask")

    if ask_button and not query.strip():
        st.warning("Please enter a question.")

    if ask_button and query.strip():
        query = query.strip()
        st.session_state.pop("qa_result", None)
        with st.spinner("Searching the document..."):
            initial_results = hybrid_retrieve_chunks(
                query=query,
                chunks=chunks,
                embedding_model=get_embedding_model(),
                chunk_embeddings=chunk_embeddings,
                top_k=8,
                lexical_top_k=12,
                semantic_top_k=12,
                lexical_min_score=0.02,
                lexical_weight=1.0,
                semantic_weight=1.0,
                rrf_k=60,
            )

            direct_answer = INSUFFICIENT_CONTEXT_MESSAGE
            support_sections = []
            if initial_results:
                reranker = get_reranker()
                reranked_results = rerank_chunks(query, initial_results, reranker, top_k=5)
                support_results = filter_and_deduplicate_supports(
                    reranked_results, max_results=3, min_rerank_score=0.0,
                )

                if support_results:
                    direct_answer = generate_answer(
                        query=query,
                        answer_mode=answer_mode,
                        ollama_model_name=ollama_model_name,
                        answer_source_results=support_results,
                        reranker=reranker,
                    )
                    support_sections = [
                        (ranked, extract_support_excerpt(query, ranked, reranker))
                        for ranked in support_results
                    ]

            st.session_state["qa_result"] = {
                "query": query,
                "answer": direct_answer,
                "answer_mode": answer_mode,
                "ollama_model_name": ollama_model_name,
                "support_sections": support_sections,
            }

    result = st.session_state.get("qa_result")
    if result is None:
        return

    st.write("### Answer")
    st.write(f"**Question:** {result['query']}")
    mode_caption = f"Mode: {result['answer_mode']}"
    if result["answer_mode"] == "Grounded LLM (Ollama)":
        mode_caption += f" | Model: {result['ollama_model_name']}"
    st.caption(mode_caption)
    st.write(result["answer"])
    render_support_sections(result["support_sections"], show_debug=show_debug)


def render_raw_text_tab(full_text: str) -> None:
    st.subheader("Extracted Text")
    st.text_area("Raw text preview", full_text[:20000], height=400)


def main() -> None:
    apply_page_style()

    show_debug, answer_mode, ollama_model_name = render_sidebar_tools()
    render_header()

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    reset_question_state(sha256(file_bytes).hexdigest() if file_bytes is not None else None)

    if uploaded_file is None:
        return

    try:
        full_text, chunk_texts = get_cached_docling_parse(
            file_bytes,
            uploaded_file.name,
        )
        chunks = rebuild_chunks(chunk_texts)
        chunk_embeddings = get_cached_chunk_embeddings(tuple(chunk_texts))
    except Exception as e:
        st.error(f"Document processing failed: {e}")
        st.stop()

    if not full_text.strip():
        st.error("No readable text was extracted from this PDF.")
        st.stop()

    st.success("Document processed successfully.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Keywords", "Ask Questions", "Raw Text"]
    )

    with tab1:
        render_summary_tab(chunk_texts, chunk_embeddings)

    with tab2:
        render_keywords_tab(full_text)

    with tab3:
        render_question_tab(
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            show_debug=show_debug,
            answer_mode=answer_mode,
            ollama_model_name=ollama_model_name,
        )

    with tab4:
        render_raw_text_tab(full_text)


if __name__ == "__main__":
    main()
