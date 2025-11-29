import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()


# Load API key safely from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(" GROQ_API_KEY not found. Please set it as an environment variable.")

class RAGAgent:
    """
    Retrieval-Augmented Generation agent using Groq LLMs.
    """

    def __init__(self, model="llama-3.1-8b-instant"):
        # Correct: pass the variable (not the string)
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model

    def generate(self, query: str, retrieved_chunks: list):
        """
        Build a RAG prompt using retrieved context and user query.
        """
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are MediSearch AI, a medical research assistant.

Your job is to answer questions ONLY using the information in the provided context.
If the context does not give enough evidence, respond:
"Insufficient information in the study."

CONTEXT:
{context}

QUESTION:
{query}

EVIDENCE-BASED ANSWER:
"""

        # Generate the answer
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content
