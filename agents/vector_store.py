

import faiss
import numpy as np
import os

class VectorStore:
    """
    Vector store for handling:
    - FAISS index creation
    - Adding embeddings
    - Semantic search
    - Saving/loading FAISS index
    """

    def __init__(self, dim=384, index_path="outputs/faiss.index"):
        """
        dim: embedding dimension (MiniLM = 384)
        index_path: where to save/load the FAISS index
        """
        self.dim = dim
        self.index_path = index_path

        # Create a flat L2 FAISS index
        self.index = faiss.IndexFlatL2(dim)

    # -----------------------------------
    # Add embeddings to FAISS index
    # -----------------------------------
    def add(self, embeddings: np.ndarray):
        """
        Accepts embeddings of shape (N, dim)
        """
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    # -----------------------------------
    # Search most similar chunks
    # -----------------------------------
    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        query_embedding: shape (1, dim)
        Returns: [indices of top_k most similar vectors]
        """
        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0]

    # -----------------------------------
    # Save FAISS index
    # -----------------------------------
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    # -----------------------------------
    # Load FAISS index
    # -----------------------------------
    def load(self):
        """
        Load FAISS index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at: {self.index_path}")

        self.index = faiss.read_index(self.index_path)

    # -----------------------------------
    # Get size of the index
    # -----------------------------------
    def count(self):
        return self.index.ntotal
