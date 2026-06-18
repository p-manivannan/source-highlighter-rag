# Citation-Grounded Source Highlighter RAG

A local-first research assistant that answers questions over PDF collections and makes every generated claim inspectable. The app combines semantic search, keyword retrieval, structured citation validation, and an interactive PDF viewer so users can move from an answer directly to the highlighted source passage that supports it.

## What It Does

- Ingests PDFs from `data/`, chunks them with source metadata, and stores embeddings in a persistent ChromaDB collection.
- Retrieves evidence with a hybrid vector + BM25 search pipeline for stronger recall across semantic and exact-match queries.
- Generates citation-grounded answers with Gemini, validates that cited evidence maps back to retrieved chunks, and renders clickable citations.
- Opens cited PDFs in a custom React/PDF.js source viewer with exact quote or fallback chunk highlighting.
- Evaluates RAG quality with Ragas metrics, expected-source coverage, answer caching, and local Ollama-based judging.

## Tech Stack

- **RAG and retrieval:** LlamaIndex, ChromaDB, HuggingFace BGE embeddings, `rank-bm25`
- **LLM and evaluation:** Gemini, Ragas, LangChain Ollama
- **UI:** Streamlit, React, TypeScript, Vite, PDF.js / React PDF
- **Runtime and tooling:** Python 3.12, uv, pytest

## Project Structure

```text
.
|-- app.py                         # Streamlit chat app and RAG orchestration
|-- ingest.py                      # PDF ingestion, chunking, embedding, Chroma persistence
|-- evaluate_rag.py                # Ragas evaluation and diagnostics
|-- source_viewer/                 # Streamlit component bridge and React PDF viewer
|   `-- frontend/
|-- data/                          # PDFs and demo evaluation questions
|-- chroma_db/                     # Local persisted vector store
`-- tests/                         # Unit tests for citation validation and evaluation logic
```

## Setup

Install dependencies:

```powershell
uv sync
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Then set:

```env
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_MODEL=models/gemini-2.5-flash
EVALUATION_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

For evaluation, install the local judge model:

```powershell
ollama pull qwen2.5:3b
```

## Run Locally

Add PDF files to `data/`, then build the local index:

```powershell
uv run python ingest.py
```

Start the app:

```powershell
uv run streamlit run app.py
```

Ask a question in the chat UI, then click any citation to inspect the supporting PDF passage.

## Evaluate

Run the RAG evaluation suite after ingestion:

```powershell
uv run python evaluate_rag.py
```

Useful options:

```powershell
uv run python evaluate_rag.py --refresh-answers
uv run python evaluate_rag.py --questions data/demo_questions.json --output evaluation_results.json
```

The evaluation report includes context precision, context recall, faithfulness, answer relevancy, answer correctness, expected-source coverage, and failure-pattern diagnostics.

## Test

```powershell
uv run pytest
```

## Resume Highlights

- Built a citation-grounded RAG system that converts PDF answers into inspectable, claim-level evidence links.
- Implemented hybrid retrieval with ChromaDB vector search, HuggingFace embeddings, and BM25 keyword scoring.
- Delivered a full-stack AI workflow with Streamlit, React/TypeScript PDF highlighting, and Ragas-based quality evaluation.
