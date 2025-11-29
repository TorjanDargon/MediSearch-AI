

import pdfplumber

class PDFExtractionAgent:

    def extract_text(self, pdf_path: str) -> str:
        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text += f"\n\n--- Page {i+1} ---\n{text}"

        return full_text
