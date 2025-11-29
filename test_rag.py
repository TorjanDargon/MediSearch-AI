from agents.RAG_synthesis import RAGAgent

# Function to load your chunk file
def load_chunks(path, separator="---"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split chunks
    return [c.strip() for c in text.split(separator) if c.strip()]


# Load your actual chunk file
chunks = load_chunks("outputs/cleaned_text.txt")   # <-- YOUR file

# Choose top 3 chunks for testing
retrieved_chunks = chunks[:3]

query = "What does the research say about esophagectomy treatment options?"

# Initialize RAG agent with Groq model
rag = RAGAgent(model="openai/gpt-oss-20b")

print("Testing Groq RAG Agent with REAL CHUNKS...\n")

answer = rag.generate(query, retrieved_chunks)

print("\n===== GROQ RAG OUTPUT =====\n")
print(answer)
