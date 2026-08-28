import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline
from src.analytics import SessionAnalytics

load_dotenv()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return EmbeddingModel()


st.set_page_config(
    page_title="ResearchInsight AI",
    page_icon="research",
    layout="wide"
)

st.title("ResearchInsight AI")
st.caption("Upload research papers and ask questions powered by RAG + Gemini")

# Session state initialization
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}
if "analytics" not in st.session_state:
    st.session_state.analytics = SessionAnalytics()


# ----- Sidebar: Upload & Process -----
with st.sidebar:
    st.header("Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs", type="primary"):
        with st.status("Processing PDFs...", expanded=True) as status:
            try:
                all_chunks = []
                processed_files = []

                for uploaded_file in uploaded_files:
                    st.write(f"Extracting text from **{uploaded_file.name}**...")
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages = extract_text_from_pdf(tmp_path)
                    os.unlink(tmp_path)

                    for page in pages:
                        page["source"] = uploaded_file.name

                    st.write(f"Chunking **{uploaded_file.name}** ({len(pages)} pages)...")
                    chunks = create_chunks(pages)
                    all_chunks.extend(chunks)
                    processed_files.append(
                        {"name": uploaded_file.name, "pages": len(pages), "chunks": len(chunks)}
                    )

                st.write("Loading embedding model...")
                embedding_model = load_embedding_model()

                st.write(f"Generating embeddings for {len(all_chunks)} chunks...")
                texts = [c["text"] for c in all_chunks]
                embeddings = embedding_model.encode_texts(texts)

                st.write("Building vector index...")
                vector_store = VectorStore(dimension=embeddings.shape[1])
                vector_store.add_embeddings(embeddings)

                retriever = Retriever(embedding_model, vector_store, all_chunks)
                st.session_state.pipeline = RAGPipeline(retriever)
                st.session_state.chat_history = []
                st.session_state.analytics = SessionAnalytics()
                st.session_state.analytics.record_documents(
                    num_docs=len(processed_files),
                    num_pages=sum(f["pages"] for f in processed_files),
                    num_chunks=len(all_chunks)
                )
                st.session_state.doc_info = {
                    "files": processed_files,
                    "total_chunks": len(all_chunks)
                }

                status.update(
                    label=f"Done — {len(processed_files)} file(s), {len(all_chunks)} chunks indexed.",
                    state="complete",
                    expanded=False,
                )

            except ValueError as e:
                status.update(label="Configuration error", state="error")
                st.error(f"Configuration error: {e}")
            except Exception as e:
                status.update(label="Processing failed", state="error")
                st.error(f"Processing failed: {e}")

    # Show indexed document info
    if st.session_state.pipeline and st.session_state.doc_info:
        st.divider()
        st.subheader("Indexed Documents")
        for f in st.session_state.doc_info["files"]:
            st.markdown(
                f"**{f['name']}** — {f['pages']} pages, {f['chunks']} chunks"
            )
        st.caption(
            f"Total chunks in index: {st.session_state.doc_info['total_chunks']}"
        )

        st.divider()
        st.subheader("Session Analytics")
        stats = st.session_state.analytics.summary()
        col1, col2 = st.columns(2)
        col1.metric("Queries", stats["total_queries"])
        col2.metric("Avg Response", f"{stats['avg_response_time']}s")
        col1.metric("Pages Indexed", stats["total_pages"])
        col2.metric("Chunks Indexed", stats["total_chunks"])

        if st.button("Clear & Reset"):
            st.session_state.pipeline = None
            st.session_state.chat_history = []
            st.session_state.doc_info = {}
            st.session_state.analytics = SessionAnalytics()
            st.rerun()


# ----- Main: Chat Interface -----
if st.session_state.pipeline is None:
    st.info("Upload one or more PDF files in the sidebar, then click 'Process PDFs' to get started.")
else:
    # Replay chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"- **{src['source']}** | Page {src['page_number']} "
                            f"| Similarity: {src['score']:.4f}"
                        )

    # Chat input
    query = st.chat_input("Ask a question about your research papers...")

    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        # Generate and display assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    st.session_state.analytics.start_query()
                    result = st.session_state.pipeline.generate_answer(query)
                    st.session_state.analytics.end_query()
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(
                                    f"- **{src['source']}** | Page {src['page_number']} "
                                    f"| Similarity: {src['score']:.4f}"
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    err_msg = str(e)
                    st.error(f"Error generating answer: {err_msg}")
                    if "API_KEY" in err_msg.upper() or "401" in err_msg or "403" in err_msg:
                        st.warning(
                            "Check that your GEMINI_API_KEY in the .env file is valid. "
                            "Get a key from https://aistudio.google.com/app/apikey"
                        )
