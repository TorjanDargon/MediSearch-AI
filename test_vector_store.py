from agents.vector_store import VectorStore
import numpy as np

emb = np.random.rand(10, 384).astype("float32")   # 10 embeddings

store = VectorStore(dim=384)
store.add(emb)
store.save()

print("Stored vectors:", store.count())

# Query test
query = np.random.rand(1, 384).astype("float32")
idxs = store.search(query, top_k=3)
print("Top matches:", idxs)
