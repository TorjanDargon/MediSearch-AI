import os
import json
from groq import Groq
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()
class HypothesisAgent:
    """
    Generates testable hypotheses from retrieved chunks using Groq LLMs.
    Returns parsed JSON (list of hypothesis dicts) when possible.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b", temperature: float = 0.2):
        """
        model: Groq model to use
        temperature: lower -> more conservative / factual outputs
        """
        self.client = Groq(api_key= os.getenv("GROQ_API_KEY"))  
        self.model = model
        self.temperature = temperature

    def _build_prompt(self, topic: str, retrieved_chunks: List[str], max_hypotheses: int = 3) -> str:
        """
        Prompt requests a JSON array of hypotheses. Each hypothesis should include:
         - hypothesis (one-sentence)
         - mechanism (short mechanistic explanation)
         - supporting_chunks (list of integers referencing the provided chunk list, 0-based)
         - confidence (0.0-1.0)
         - suggested_experiments (bullet list, plain strings)
        """
        context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(retrieved_chunks))

        prompt = f"""
You are a biomedical research assistant. Using ONLY the evidence in CONTEXT, produce up to {max_hypotheses} scientifically plausible, testable hypotheses about: "{topic}".

CONTEXT (each chunk is numbered):
{context}

INSTRUCTIONS:
1) Output a JSON array (no surrounding explanation) with up to {max_hypotheses} objects.
2) Each object must have the keys:
   - "hypothesis": a concise testable statement (<= 35 words).
   - "mechanism": plausible mechanistic explanation (1-3 sentences).
   - "supporting_chunks": list of integers (0-based) pointing to context chunks that support this hypothesis.
   - "confidence": number between 0.0 and 1.0 estimating how well supported it is by the context.
   - "suggested_experiments": list of 1-3 practical experiment ideas or analyses to test the hypothesis.

3) If you cannot produce hypotheses supported by the context, return an empty JSON array: [].
4) Be conservative; do NOT hallucinate citations beyond the provided chunks.
5) This is NOT medical advice — output is for research brainstorming only.

Output ONLY the JSON array.
"""
        return prompt

    def generate(self, topic: str, retrieved_chunks: List[str], max_hypotheses: int = 3) -> List[Dict[str, Any]]:
        """
        Generate hypotheses and return parsed JSON (list of dicts).
        If parsing fails, returns a single-element list with a 'raw_output' field.
        """
        prompt = self._build_prompt(topic, retrieved_chunks, max_hypotheses=max_hypotheses)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024
        )

        # Groq returns message object; access .content
        text = resp.choices[0].message.content

        # Try to parse JSON. Sometimes model adds code fences — strip them.
        try:
            # remove common wrappers/triple backticks
            cleaned = text.strip()
            if cleaned.startswith("```"):
                # strip triple backticks and optional language marker
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            # Attempt to load JSON
            parsed = json.loads(cleaned)
            # Validate structure (list of dicts)
            if isinstance(parsed, list):
                return parsed
            else:
                # wrap in list if single dict returned
                if isinstance(parsed, dict):
                    return [parsed]
        except Exception as e:
            # parsing failed — return fallback
            return [{"raw_output": text, "parse_error": str(e)}]
