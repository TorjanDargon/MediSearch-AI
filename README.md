# MediSearch AI 🧬🔬

> **An agentic AI research assistant that reads, understands, and reasons over medical literature — and generates novel hypotheses for medical discovery.**

---

## Overview

MediSearch AI is a Python-based, end-to-end agentic pipeline for medical research. Upload any medical research paper in PDF format, ask a question, and the system will:

1. **Extract** raw text from the PDF
2. **Clean and chunk** the text intelligently
3. **Embed** chunks using semantic sentence embeddings
4. **Index** them in a FAISS vector store for fast similarity search
5. **Retrieve** the most relevant passages for your query
6. **Answer** your question using a RAG (Retrieval-Augmented Generation) agent
7. **Generate hypotheses** for further medical discovery based on the paper's content

All of this is exposed via a clean **Streamlit web UI** and can also be run directly from the **command line**.

---

## Features

- 📄 **PDF Ingestion** — Upload any medical research paper (PDFPlumber / PyMuPDF)
- 🧹 **Text Cleaning & Chunking** — Configurable chunk sizes for precise retrieval
- 🧠 **Semantic Embeddings** — Uses `sentence-transformers` for dense vector representations
- ⚡ **FAISS Vector Search** — Millisecond-level similarity retrieval over large documents
- 💬 **RAG Q&A** — Contextually grounded answers from an LLM
- 🔬 **Hypothesis Generation** — AI-driven novel research hypothesis proposals
- 🖥️ **Streamlit UI** — Interactive web interface for non-technical users
- 🖱️ **CLI Support** — Full pipeline accessible from the terminal
- 💾 **Output Persistence** — Saves extracted text, chunks, embeddings, FAISS index, and a JSON run summary

---

## Project Structure

```
MediSearch-AI/
├── agents/
│   ├── pdf_extract.py        # PDFExtractionAgent — extracts raw text from PDFs
│   ├── pdf_cleaner.py        # TextCleanerChunker — cleans and chunks text
│   ├── embedding.py          # EmbeddingAgent — generates sentence embeddings
│   ├── vector_store.py       # VectorStore — manages FAISS index (save/load/add)
│   ├── vector_retriever.py   # Retriever — top-k similarity search
│   ├── RAG_synthesis.py      # RAGAgent — LLM-based answer generation
│   └── hypothesis.py         # HypothesisAgent — novel hypothesis generation
├── outputs/                  # Auto-generated: embeddings, FAISS index, run summaries
├── papers/                   # Place your PDF files here
├── app.py                    # Streamlit web application
├── pipeline.py               # Core pipeline orchestrator (also a CLI entrypoint)
├── requirements.txt          # Python dependencies
├── test_chunker.py           # Unit test — text chunker
├── test_extractor.py         # Unit test — PDF extractor
├── test_hypothesis.py        # Unit test — hypothesis agent
├── test_rag.py               # Unit test — RAG agent
├── test_retriever.py         # Unit test — retriever
└── test_vector_store.py      # Unit test — vector store
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/TorjanDargon/MediSearch-AI.git
cd MediSearch-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` will install a CPU build by default. If you have a GPU, follow [PyTorch's installation guide](https://pytorch.org/get-started/locally/) to install the appropriate CUDA version.

### 4. Set up environment variables

Create a `.env` file in the project root and add your LLM API key:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

### Option A — Streamlit Web UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. From there you can:

- Upload a PDF research paper
- Enter a query about the paper
- Optionally specify a topic for hypothesis generation
- Tune chunk size and number of retrieved passages
- Click **Run Pipeline** to get your answer and generated hypotheses

### Option B — Command Line

```bash
python pipeline.py \
  --pdf papers/your_paper.pdf \
  --query "What does this paper say about metformin?" \
  --topic "metformin and type 2 diabetes" \
  --chunk-size 300 \
  --top-k 3
```

Add `--force-reindex` to recompute embeddings if you've changed the PDF or chunk settings.

### Option C — Python API

```python
from pipeline import Pipeline

p = Pipeline()
result = p.process_pdf_and_answer(
    pdf_path="papers/sample.pdf",
    query="What are the key findings on drug resistance?",
    topic_for_hypotheses="antibiotic resistance mechanisms",
    chunk_size=300,
    top_k=3,
)

print(result["answer"])
print(result["hypotheses"])
```

---

## Pipeline Architecture

```
PDF File
   │
   ▼
PDFExtractionAgent       ← extracts raw text
   │
   ▼
TextCleanerChunker        ← cleans noise, splits into word-level chunks
   │
   ▼
EmbeddingAgent            ← sentence-transformers (384-dim vectors)
   │
   ▼
VectorStore (FAISS)       ← indexed and persisted to disk
   │
   ▼
Retriever                 ← top-k semantic search for the query
   │
   ├──▶ RAGAgent          ← generates grounded answer via LLM
   │
   └──▶ HypothesisAgent   ← generates novel research hypotheses
```

---

## Configuration

The `Pipeline` class accepts several configuration parameters:

| Parameter | Default | Description |
|---|---|---|
| `outputs_dir` | `"outputs"` | Directory for all saved outputs |
| `embedding_dim` | `384` | Embedding vector dimension |
| `faiss_top_k` | `3` | Default number of chunks to retrieve |
| `rag_model` | `"openai/gpt-oss-20b"` | LLM used for answer generation |
| `hypo_model` | `"openai/gpt-oss-20b"` | LLM used for hypothesis generation |
| `chunk_size` | `300` | Words per text chunk |
| `force_reindex` | `False` | Recompute embeddings even if index exists |

---

## Output Files

After a successful run, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `extracted_text.txt` | Raw text extracted from the PDF |
| `cleaned_text.txt` | Cleaned and normalized text |
| `chunks.txt` | Text chunks separated by `---` |
| `embeddings.npy` | NumPy array of sentence embeddings |
| `faiss.index` | FAISS vector index (reusable across runs) |
| `last_run_summary.json` | Full JSON summary of the last pipeline run |

---

## Running Tests

Each agent has a corresponding test file. Run them individually or all at once:

```bash
# Run all tests
python -m pytest test_*.py -v

# Run a specific test
python test_rag.py
```

---

## Dependencies

Key libraries used:

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pdfplumber` / `pymupdf` | PDF text extraction |
| `sentence-transformers` | Semantic embeddings |
| `faiss-cpu` | Vector similarity search |
| `langchain` / `langgraph` | Agent orchestration |
| `openai` | LLM API calls |
| `spacy` / `scispacy` | Biomedical NLP |
| `torch` | ML model inference |
| `pandas` / `plotly` | Data utilities |

---

## Roadmap

- [ ] Support for multi-PDF ingestion and cross-document reasoning
- [ ] PubMed / arXiv live paper search integration
- [ ] Knowledge graph construction (Neo4j)
- [ ] Clinical trial data analysis support
- [ ] Export results to PDF/Word report
- [ ] Docker container for one-command deployment

---

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes, then submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is open source. Please add a `LICENSE` file to clarify the terms of use.

---

## Author

**TorjanDargon** — [GitHub](https://github.com/TorjanDargon)

---

> ⚠️ **Disclaimer:** MediSearch AI is a research tool and is not intended for clinical diagnosis or medical decision-making. Always consult qualified medical professionals for health-related decisions.
