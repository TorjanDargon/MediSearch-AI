

import re
from typing import List

class TextCleanerChunker:

    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text:
        - Remove multiple newlines
        - Remove extra spaces
        - Remove weird characters
        """
        text = re.sub(r'\n+', '\n', text)                # remove excessive newlines
        text = re.sub(r'\s+', ' ', text)                 # collapse spaces
        text = text.replace("\x0c", " ")                 # remove form feed
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        """
        Split into chunks of ~800 words.
        """
        words = text.split(" ")
        chunks = []

        current = []
        count = 0

        for w in words:
            current.append(w)
            count += 1

            if count >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                count = 0

        if current:
            chunks.append(" ".join(current))

        return chunks
