from agents.pdf_cleaner import TextCleanerChunker

cleaner = TextCleanerChunker()

sample_text = open("outputs/extracted_text.txt", encoding="utf-8").read()

clean = cleaner.clean_text(sample_text)
chunks = cleaner.chunk_text(clean, chunk_size=300)
with open("outputs/cleaned_text.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("Cleaned length:", len(clean))
print("Chunks:", len(chunks))
print("Preview chunk 1:", chunks[0][:500])
