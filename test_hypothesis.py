from agents.hypothesis import HypothesisAgent

# Load your chunks from file (example)
def load_chunks(path="outputs/cleaned_text.txt", separator="---"):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return [c.strip() for c in txt.split(separator) if c.strip()]

chunks = load_chunks("outputs/cleaned_text.txt")  # or outputs/cleaned_text.txt depending on your format
topic = "The risk analysis of unplanned tracheal intubation after radical esophagectomy"

agent = HypothesisAgent(model="openai/gpt-oss-20b", temperature=0.2)
hypotheses = agent.generate(topic, chunks[:8], max_hypotheses=3)  # send top N chunks or FAISS-retrieved chunks

print("Generated hypotheses:")
for i, h in enumerate(hypotheses):
    print(f"\nHypothesis #{i+1}:")
    if "raw_output" in h:
        print("RAW OUTPUT (could not parse JSON):")
        print(h["raw_output"])
        print("Parse error:", h.get("parse_error"))
    else:
        print("Hypothesis:", h.get("hypothesis"))
        print("Mechanism:", h.get("mechanism"))
        print("Supporting chunks:", h.get("supporting_chunks"))
        print("Confidence:", h.get("confidence"))
        print("Suggested experiments:", h.get("suggested_experiments"))
