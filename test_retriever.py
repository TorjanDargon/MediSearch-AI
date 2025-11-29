'''from agents.vector_retriever import Retriever
from agents.vector_store import VectorStore
from agents.embedding import EmbeddingAgent
import numpy as np

# Load fake chunks
chunks = [
    "Diabetes is managed with insulin.",
    "Exercise improves glucose control.",
    "Metformin is commonly used.",
]

# Fake embeddings (for testing only)
embeddings = np.random.rand(3, 384).astype("float32")
np.save("outputs/test_embeddings.npy", embeddings)

embedder = EmbeddingAgent()
store = VectorStore(dim=384)
store.add(embeddings)

retriever = Retriever(store, chunks, embedder)

query = "What medicine helps diabetes?"

results = retriever.retrieve(query)

print("\nRetrieved Chunks:")
for r in results:
    print("-", r)
'''

from agents.embedding import EmbeddingAgent
from agents.vector_store import VectorStore
from agents.vector_retriever import Retriever
import numpy as np

def load_chunks_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split('---') if chunk.strip()]
    return chunks

chunk_file_path = "outputs/cleaned_text.txt"
chunks = load_chunks_from_file(chunk_file_path)

embedder = EmbeddingAgent()
emb = embedder.embed(chunks)  # Replace with your embedder's actual method to get embeddings

store = VectorStore(dim=emb.shape[1])
store.add(emb)

retriever = Retriever(store, chunks, embedder)

query = "What is esophagectomy?"
results = retriever.retrieve(query)

print(results)

