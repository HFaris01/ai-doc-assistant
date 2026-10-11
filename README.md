# AI Doc Assistant

AI Doc Assistant is a Streamlit application for asking questions about PDFs that contain text.

The app reads an uploaded PDF and searches for passages related to your question. It also provides a summary made from document excerpts and a list of frequent keywords. You can choose between two answer modes:

- **Extractive:** selects an answer passage from text that passes the evidence filter.
- **Grounded LLM (Ollama):** sends the filtered passages to an Ollama model and asks it to answer using that text.

You can read the retrieved passages below each answer. If none pass the evidence filter, the app reports that it needs more information. Accepted passages may still be irrelevant, and generated answers can contain errors.

## Features

- Upload and process PDF documents
- Summaries made from document excerpts
- Keyword extraction
- Hybrid retrieval
  - TF-IDF lexical retrieval
  - SentenceTransformer semantic embedding retrieval
- Cross-encoder reranking
- Two answer modes
  - Extractive
  - Grounded LLM (Ollama)
- Retrieved passages and optional retrieval diagnostics
- Answers remain visible when you change options
- Local Ollama answering by default

## Libraries

- Frontend / App: Streamlit
- PDF parsing: Docling
- Lexical retrieval: TF-IDF (scikit-learn)
- Semantic retrieval: SentenceTransformer (all-MiniLM-L6-v2)
- Reranking: CrossEncoder (cross-encoder/ms-marco-MiniLM-L6-v2)
- Answer generation: Ollama
- NLP utilities: NLTK, NumPy, scikit-learn

## How it works

1. A PDF is uploaded through the Streamlit interface.
2. The document is parsed with Docling and split into chunks.
3. Chunk embeddings are generated with a SentenceTransformer model.
4. For a user question, the app retrieves relevant chunks using:
   - lexical TF-IDF similarity
   - semantic embedding similarity
5. The retrieved chunks are fused and reranked using a cross-encoder.
6. The app removes passages that fail the evidence filter, along with duplicates. If none remain, it reports that it needs more information.
7. The app uses accepted passages in one of two ways:
   - Extractive: selects a passage and limits its length for display.
   - Grounded LLM: asks the selected Ollama model to answer using the supplied context.
8. The app shows excerpts around the selected evidence and saves the result for the current session. Changing answer options applies to the next question. Replacing or removing the PDF clears the saved result.

## Installation

The application requires the packages in `requirements.txt`. Their versions are currently unpinned, so installation results can vary over time. The regression tests described below use only Python's standard library.

First use may download Docling and SentenceTransformer model files and NLTK resources. Allow internet access for these downloads, or prepare the required resources beforehand. Ollama models are downloaded separately. Larger documents and models can take longer to process and need more memory. Performance also depends on your hardware.

### 1. Clone the repository

Run:

    git clone https://github.com/HFaris01/ai-doc-assistant.git
    cd ai-doc-assistant

### 2. Create and activate a virtual environment

In Windows PowerShell, run:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### 3. Install dependencies

Run:

    python -m pip install -r requirements.txt

### 4. Set up Ollama (optional for extractive mode)

For **Grounded LLM (Ollama)** mode, install [Ollama](https://ollama.com/) and make sure its service is running. Skip this step if you intend to use only **Extractive** mode.

Download the default model:

    ollama pull gemma3:latest

You can verify it with:

    ollama list

Enter an installed model's exact name in the app sidebar. The default is `gemma3:latest`. Model tags can change over time.

### 5. Run the app

Start the Streamlit app with:

    streamlit run app.py

The app initially selects **Grounded LLM (Ollama)**. Choose **Extractive** in the sidebar if you have not set up Ollama.

## Configuration and data handling

- `DOCLING_ARTIFACTS_PATH`: optional path to local Docling model artifacts.
- `OLLAMA_HOST`: Ollama endpoint, defaulting to `http://localhost:11434`.
- **Ollama model name** in the sidebar: model to use for the next LLM answer.

Document processing and retrieval run on the machine hosting Streamlit. LLM mode sends your question and the accepted passages to the configured Ollama endpoint. If `OLLAMA_HOST` points to another machine, those requests leave the Streamlit host.

Streamlit caches parsed text and embeddings. Removing a PDF clears the saved question and answer. The processing results remain cached and can be cleared under **Developer Tools**. Reloading the browser or restarting the server clears saved answers.

## Answer modes

### Extractive

This mode selects a passage from accepted document text and shortens it for display. It works without a running Ollama service. If no usable passage remains, the app reports that it needs more information.

### Grounded LLM (Ollama)

This mode sends up to three accepted passages to the selected model through Ollama and asks it to answer using that text. The model can produce unsupported claims. Check its answer against the retrieved passages.

Answer accuracy depends on retrieval quality and the selected model. The two modes have yet to be compared in a benchmark.

## Known limitations

- OCR is disabled. Scanned PDFs without a text layer are not supported by the current configuration.
- Text processing and retrieval are primarily aimed at English documents.
- Evidence selection uses heuristic thresholds and text filters. These can admit irrelevant passages or reject useful ones.
- Long chunks can exceed the embedding model's input limit. Chunk sizes still need to be adjusted to that limit.
- Retrieved passages do not yet include PDF page numbers or section citations.
- Documents with many tables may produce overly broad answers.
- Extractive mode can return incomplete answers to questions asking for definitions.
- Summary quality can vary depending on document structure.
- Passages taken from tables may need better handling of rows and headings.

## Regression tests

Run the focused tests from the repository root:

    python -m unittest discover -s tests -v

These tests use Python's standard library and were checked with Python 3.12. They use simulated library and model responses to run the application code. No dependency installation or model download is required.

The tests check evidence filtering and answer persistence. They also check how selected text is shortened for display. PDF parsing and model output require integration testing. Browser behavior requires separate checks.

## Planned improvements

- Improve answers to questions about tables.
- Use document sections when selecting summary passages.
- Improve passage filtering for specific types of questions.
- Add a dataset for measuring answer accuracy.
- Add an optional cloud LLM mode.

## About the project

This portfolio project implements a PDF question answering workflow using document retrieval and local model inference.
