# AI Doc Assistant

AI Doc Assistant is a Streamlit application for asking questions about PDF documents.

The app parses uploaded PDFs, generates a summary, extracts keywords, retrieves relevant document chunks using hybrid lexical and semantic retrieval, reranks them with a cross-encoder, and answers questions in two modes:

- Extractive: selects an answer from the retrieved text
- Grounded LLM: sends the top chunks to a local Ollama model and generates a grounded answer

Supporting sections are displayed below the answer for transparency.

## Features

- Upload and process PDF documents
- Automatic summary generation
- Keyword extraction
- Hybrid retrieval
  - TF-IDF lexical retrieval
  - SentenceTransformer semantic embedding retrieval
- Cross-encoder reranking
- Two answer modes
  - Extractive
  - Grounded LLM (Ollama)
- Supporting document sections shown for transparency
- Local-first workflow for LLM answering

## Tech Stack

- Frontend / App: Streamlit
- PDF parsing: Docling
- Lexical retrieval: TF-IDF (scikit-learn)
- Semantic retrieval: SentenceTransformer (all-MiniLM-L6-v2)
- Reranking: CrossEncoder (cross-encoder/ms-marco-MiniLM-L6-v2)
- Local grounded answering: Ollama
- NLP utilities: NLTK, NumPy, Pandas, scikit-learn

## How It Works

1. A PDF is uploaded through the Streamlit interface.
2. The document is parsed with Docling and split into chunks.
3. Chunk embeddings are generated with a SentenceTransformer model.
4. For a user question, the app retrieves relevant chunks using:
   - lexical TF-IDF similarity
   - semantic embedding similarity
5. The retrieved chunks are fused and reranked using a cross-encoder.
6. The app answers in one of two ways:
   - Extractive: selects an answer from the retrieved text
   - Grounded LLM: sends the top chunks to a local Ollama model and generates an answer only from that context
7. Supporting sections are displayed below the answer.

## Installation

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

    pip install -r requirements.txt

### 4. Install and run Ollama

Install Ollama from the official website.

Then pull a local model, for example:

    ollama pull gemma3:latest

You can verify it with:

    ollama list

### 5. Run the app

Start the Streamlit app with:

    streamlit run app.py

## Answer Modes

### Extractive

This mode returns an answer directly from the retrieved document text.

### Grounded LLM (Ollama)

This mode sends the top retrieved chunks to a local LLM through Ollama and asks it to answer only from the provided context.

This mode generally produces more natural, concise, and often more accurate answers than extractive mode, especially for definition-style questions.

## Example Use Cases

- Ask questions about lecture slides
- Ask questions about research papers
- Summarize uploaded PDFs
- Extract key terms from technical documents
- Compare extractive QA vs grounded LLM QA on the same document

## Current Limitations

- Table-heavy documents may produce overly broad answers
- Extractive mode is weak on definition-style questions
- Summary quality can vary depending on document structure
- Table-derived chunks may need more specialized filtering or table-aware processing

## Future Improvements

- Better table-aware question answering
- Stronger section-aware summarization
- More precise filtering for application-specific questions
- Evaluation set for systematic benchmarking
- Optional cloud LLM mode in addition to local Ollama mode

## Project Goal

This project was built as a portfolio piece to demonstrate practical AI engineering skills in:

- document parsing
- NLP retrieval
- neural reranking
- local LLM integration
- user-facing application development with Streamlit