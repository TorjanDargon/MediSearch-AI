'''from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load chunks from a text file
class EmbeddingAgent:
    def load_chunks(path: str, separator: str = None) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if separator:  
            return [c.strip() for c in text.split(separator) if c.strip()]
        else:  
            return [line.strip() for line in text.splitlines() if line.strip()]


    def embed_chunks(chunks: List[str]) -> np.ndarray:
        if not chunks:
            return np.array([])

        embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )
        return embeddings


# 🔥 CORRECT PATH HERE
    chunks = load_chunks("outputs/cleaned_text.txt", separator="---")

    embeddings = embed_chunks(chunks)

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
'''

# agents/embedding_agent.py

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingAgent:
    def __init__(self):
        # Load embedding model ONCE
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts):
        """
        Returns embeddings as a numpy array.
        Accepts list of strings.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
