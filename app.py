# streamlit application
# app.py
import streamlit as st
from PIL import Image

from config import UPLOADS_DIR
from src.rag.pipeline import ingest_pdf, ingest_image, answer_question
from src.vectorstore.chroma_store import collection_count
from src.models.ollama_client import is_ollama_running, OllamaUnavailableError, OLLAMA_NOT_RUNNING_MESSAGE

st.set_page_config(page_title="Insight Lens", layout="wide")

st.title("Insight Lens")
st.caption("Offline vision analyzer. Zero data leaves this laptop.")

if not is_ollama_running():
    st.error("Ollama is not running. Start the Ollama app, then reload this page.")
    st.stop()

# --- Sidebar: ingestion ---
with st.sidebar:
    st.header("Add a document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        save_path = UPLOADS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Ingest this file"):
            with st.spinner("Reading the document locally..."):
                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        count = ingest_pdf(str(save_path))
                    else:
                        count = ingest_image(str(save_path))
                except ConnectionError:
                    st.error(OLLAMA_NOT_RUNNING_MESSAGE)
                    st.stop()
            st.success(f"Indexed {count} page(s).")

    st.divider()
    st.metric("Pages indexed", collection_count())

# --- Main area: question and answer ---
question = st.text_input("Ask a question about your documents")

if question:
    with st.spinner("Searching locally and reasoning over the result..."):
        try:
            result = answer_question(question)
        except OllamaUnavailableError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    if result["source_image_path"]:
        st.subheader("Source")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption(f"{result['source_file']}, page {result['source_page']}")
        with col2:
            st.image(Image.open(result["source_image_path"]), use_container_width=True)