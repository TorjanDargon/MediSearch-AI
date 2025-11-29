import streamlit as st
from pipeline import Pipeline
import json

st.set_page_config(page_title="MediSearch AI", layout="wide")

st.title("MediSearch AI — Research Paper Q&A + Hypothesis Generation")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

query = st.text_input("Enter your query about the paper:")

topic = st.text_input("Optional: Enter topic for hypothesis generation (defaults to query)")

chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=1000, value=300, step=50)

top_k = st.number_input("Number of top chunks to retrieve", min_value=1, max_value=10, value=3)

force_reindex = st.checkbox("Force reindex embeddings?", value=False)

if st.button("Run Pipeline"):

    if not uploaded_pdf:
        st.warning("Please upload a PDF file to proceed.")
    elif not query.strip():
        st.warning("Please enter a query to proceed.")
    else:
        # Save uploaded PDF temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        pipeline = Pipeline()

        with st.spinner("Processing PDF and running pipeline..."):
            result = pipeline.process_pdf_and_answer(
                pdf_path="temp_uploaded.pdf",
                query=query,
                topic_for_hypotheses=topic if topic.strip() else None,
                chunk_size=chunk_size,
                top_k=top_k,
                force_reindex=force_reindex,
            )

        st.success("Pipeline run completed!")

        st.subheader("Final Answer")
        st.write(result["answer"])

        st.subheader("Generated Hypotheses")
        st.json(result["hypotheses"])

        st.subheader(f"Top {top_k} Retrieved Chunks")
        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)
            st.markdown("---")

        st.subheader("Saved File Paths")
        for key, path in result["paths"].items():
            st.write(f"{key}: `{path}`")
