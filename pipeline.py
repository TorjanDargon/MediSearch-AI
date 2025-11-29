# agents/pipeline.py
"""
Pipeline Orchestrator for MediSearch AI.

Usage:
    >>> from agents.pipeline import Pipeline
    >>> p = Pipeline()
    >>> result = p.process_pdf_and_answer("papers/sample.pdf", "What does this paper say about metformin?")
    >>> print(result["answer"])
    >>> print(result["hypotheses"])

Output (result) fields:
- answer: final RAG answer (string)
- retrieved_chunks: list of text chunks used for RAG (strings)
- hypotheses: list (parsed) from HypothesisAgent
- paths: dictionary of saved file paths (extracted, cleaned, chunks, embeddings, faiss)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

# Import your agents here.
# If your filenames/classes differ, update these imports accordingly.
from agents.pdf_extract import PDFExtractionAgent
from agents.pdf_cleaner import TextCleanerChunker
from agents.embedding import EmbeddingAgent
from agents.vector_store import VectorStore
from agents.vector_retriever import Retriever
from agents.RAG_synthesis import RAGAgent
from agents.hypothesis import HypothesisAgent

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

class Pipeline:
    def __init__(
        self,
        outputs_dir: str = "outputs",
        faiss_index_path: str = "outputs/faiss.index",
        embeddings_path: str = "outputs/embeddings.npy",
        chunks_path: str = "outputs/chunks.txt",
        cleaned_path: str = "outputs/cleaned_text.txt",
        extracted_path: str = "outputs/extracted_text.txt",
        embedding_dim: int = 384,
        faiss_top_k: int = 3,
        rag_model: str = "openai/gpt-oss-20b",
        hypo_model: str = "openai/gpt-oss-20b",
    ):
        self.outputs_dir = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)

        # paths
        self.faiss_index_path = faiss_index_path
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path
        self.cleaned_path = cleaned_path
        self.extracted_path = extracted_path

        # components
        self.pdf_extractor = PDFExtractionAgent()
        self.cleaner = TextCleanerChunker()
        self.embedder = EmbeddingAgent()
        self.vector_store = VectorStore(dim=embedding_dim, index_path=self.faiss_index_path)
        self.retriever = None  # created once chunks + index available
        self.rag_agent = RAGAgent(model=rag_model)
        self.hypo_agent = HypothesisAgent(model=hypo_model)
        self.embedding_dim = embedding_dim
        self.faiss_top_k = faiss_top_k

    # --------------------
    # Utilities
    # --------------------
    def _save_text(self, text: str, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved text to %s", path)

    def _load_chunks_file(self, path: str, separator: str = "\n---\n") -> List[str]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        # if separator present, split by it, else split by double newline or lines
        if separator and separator in txt:
            return [c.strip() for c in txt.split(separator) if c.strip()]
        if "\n\n" in txt:
            items = [t.strip() for t in txt.split("\n\n") if t.strip()]
            return items
        return [line.strip() for line in txt.splitlines() if line.strip()]

    # --------------------
    # Pipeline steps
    # --------------------
    def extract_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF and save to outputs."""
        logger.info("Extracting PDF: %s", pdf_path)
        raw_text = self.pdf_extractor.extract_text(pdf_path)
        self._save_text(raw_text, self.extracted_path)
        return raw_text

    def clean_and_chunk(self, raw_text: str, chunk_size: int = 300) -> List[str]:
        """Clean extracted text and create chunks. Save cleaned and chunks files."""
        logger.info("Cleaning text and chunking (chunk size=%d)", chunk_size)
        cleaned = self.cleaner.clean_text(raw_text)
        self._save_text(cleaned, self.cleaned_path)

        chunks = self.cleaner.chunk_text(cleaned, chunk_size=chunk_size)
        # save with separator so loaders can parse
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            f.write("\n---\n".join(chunks))
        logger.info("Saved %d chunks to %s", len(chunks), self.chunks_path)
        return chunks

    def embed_and_index(self, chunks: List[str], force_reindex: bool = False) -> Any:
        """
        Produce embeddings and add to FAISS index.
        - If a FAISS index exists and force_reindex is False, load it instead of rebuilding.
        - Returns embeddings numpy array.
        """
        # If FAISS exists and not forcing reindex, try load
        if os.path.exists(self.faiss_index_path) and not force_reindex:
            try:
                logger.info("Loading existing FAISS index from %s", self.faiss_index_path)
                self.vector_store.load()
                # Try to load embeddings if available
                if os.path.exists(self.embeddings_path):
                    import numpy as np
                    embeddings = np.load(self.embeddings_path)
                    logger.info("Loaded embeddings from %s", self.embeddings_path)
                    return embeddings
                # If embeddings not present, still ok — we can proceed without returning embeddings
                return None
            except Exception as e:
                logger.warning("Failed to load existing FAISS index: %s — will reindex. Error: %s", self.faiss_index_path, e)

        # Compute embeddings
        logger.info("Embedding %d chunks...", len(chunks))
        embeddings = self.embedder.embed(chunks)
        # save embeddings
        import numpy as np
        np.save(self.embeddings_path, embeddings)
        logger.info("Saved embeddings to %s", self.embeddings_path)

        # (re)create FAISS index and add embeddings
        logger.info("Creating FAISS index (dim=%d)...", embeddings.shape[1])
        self.vector_store = VectorStore(dim=embeddings.shape[1], index_path=self.faiss_index_path)
        self.vector_store.add(embeddings)
        self.vector_store.save()
        logger.info("FAISS index created and saved to %s", self.faiss_index_path)
        return embeddings

    def prepare_retriever(self, chunks: List[str]):
        """Instantiate retriever (requires vector_store + chunks + embedder)."""
        logger.info("Preparing retriever with %d chunks", len(chunks))
        self.retriever = Retriever(self.vector_store, chunks, self.embedder)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Return top-k relevant chunks for query (list of strings)."""
        if self.retriever is None:
            raise RuntimeError("Retriever not prepared. Call prepare_retriever() first.")
        k = top_k or self.faiss_top_k
        logger.info("Retrieving top %d chunks for query.", k)
        return self.retriever.retrieve(query, top_k=k)

    def generate_answer(self, query: str, retrieved_chunks: List[str]) -> str:
        """Use RAG agent to generate final answer."""
        logger.info("Generating RAG answer (model=%s)", self.rag_agent.model)
        return self.rag_agent.generate(query, retrieved_chunks)

    def generate_hypotheses(self, topic: str, retrieved_chunks: List[str], max_hypotheses: int = 3):
        """Use Hypothesis agent to generate hypotheses (parsed JSON if possible)."""
        logger.info("Generating up to %d hypotheses (model=%s)", max_hypotheses, self.hypo_agent.model)
        return self.hypo_agent.generate(topic, retrieved_chunks, max_hypotheses=max_hypotheses)

    # --------------------
    # High-level orchestrator
    # --------------------
    def process_pdf_and_answer(
        self,
        pdf_path: str,
        query: str,
        topic_for_hypotheses: Optional[str] = None,
        chunk_size: int = 300,
        top_k: int = 3,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Full end-to-end run:
         - Extract -> Clean/Chunk -> Embed/Index -> Retrieve -> RAG -> Hypotheses
        Returns a dictionary with answer, retrieved chunks, hypotheses, and saved paths.
        """
        logger.info("Starting pipeline for PDF: %s", pdf_path)

        # 1. Extract
        raw_text = self.extract_pdf(pdf_path)

        # 2. Clean + Chunk
        chunks = self.clean_and_chunk(raw_text, chunk_size=chunk_size)

        # 3. Embedding + Indexing
        embeddings = self.embed_and_index(chunks, force_reindex=force_reindex)

        # 4. Prepare retriever
        self.prepare_retriever(chunks)

        # 5. Retrieve
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        # 6. Answer (RAG)
        answer = self.generate_answer(query, retrieved_chunks)

        # 7. Hypotheses (optional)
        topic = topic_for_hypotheses or query
        hypotheses = self.generate_hypotheses(topic, retrieved_chunks, max_hypotheses=3)

        # 8. Save a JSON result summary
        result = {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "hypotheses": hypotheses,
            "paths": {
                "extracted": os.path.abspath(self.extracted_path),
                "cleaned": os.path.abspath(self.cleaned_path),
                "chunks": os.path.abspath(self.chunks_path),
                "embeddings": os.path.abspath(self.embeddings_path),
                "faiss_index": os.path.abspath(self.faiss_index_path),
            }
        }

        summary_path = os.path.join(self.outputs_dir, "last_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Saved pipeline summary to %s", summary_path)

        return result


# --------------------
# CLI test entrypoint
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full MediSearch AI pipeline on a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--query", required=True, help="Query to ask about the PDF")
    parser.add_argument("--topic", required=False, help="Topic (for hypothesis generation)", default=None)
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size (words)")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k chunks to retrieve")
    parser.add_argument("--force-reindex", action="store_true", help="Force recomputing embeddings and reindexing")
    args = parser.parse_args()

    pipeline = Pipeline()
    out = pipeline.process_pdf_and_answer(
        args.pdf,
        args.query,
        topic_for_hypotheses=args.topic,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        force_reindex=args.force_reindex,
    )

    print("\n=== FINAL ANSWER ===\n")
    print(out["answer"])
    print("\n=== HYPOTHESES ===\n")
    print(json.dumps(out["hypotheses"], indent=2, ensure_ascii=False))
