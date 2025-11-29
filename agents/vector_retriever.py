
'''
class Retriever:
    """
    Retrieves the top-k most relevant text chunks
    based on vector similarity search using FAISS.
    """

    def __init__(self, vector_store, chunks, embedder):
        """
        vector_store: instance of VectorStore
        chunks: list of text chunks corresponding to embeddings
        embedder: instance of EmbeddingAgent
        """
        self.vector_store = vector_store
        self.chunks = chunks
        self.embedder = embedder

    # ---------------------------------------
    # Retrieve relevant chunks for a query
    # ---------------------------------------
    def retrieve(self, query: str, top_k=3):
        """
        Returns top-k text chunks most relevant to the query.
        """
        # 1) Convert query to embedding
        query_embedding = self.embedder.embed([query])

        # 2) Search FAISS index for closest chunks
        top_indices = self.vector_store.search(query_embedding, top_k=top_k)

        # 3) Return the actual chunk texts
        results = [self.chunks[i] for i in top_indices]

        return results
'''


class Retriever:
    def __init__(self, vector_store, chunks, embedder):
        self.vector_store = vector_store
        self.chunks = chunks
        self.embedder = embedder  # EmbeddingAgent instance

    def retrieve(self, query: str, top_k=3):
        query_embedding = self.embedder.embed([query])
        indices = self.vector_store.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices]
