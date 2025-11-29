from agents.pdf_extract import PDFExtractionAgent

agent = PDFExtractionAgent()

pdf_path = "papers/sample.pdf"

print("Extracting text from:", pdf_path)
text = agent.extract_text(pdf_path)
with open("outputs/extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(text)


print("\n=== EXTRACTION PREVIEW ===\n")
print(text[:1000])    # show first 1000 characters

print("\n=== EXTRACTION LENGTH ===")
print(len(text))

