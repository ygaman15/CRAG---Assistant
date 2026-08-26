import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_engine import HybridCRAGPipeline

load_dotenv()

def extract_pdf_documents(pdf_files):
    """Extracts raw text page-by-page from uploaded PDF files."""
    documents = []
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": pdf.name, "page": page_idx + 1}
                        )
                    )
        except Exception as e:
            st.toast(f"Error parsing PDF {pdf.name}: {str(e)}", icon="⚠️")
    return documents

def create_chunks(documents):
    """Splits document text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def main():
    st.set_page_config(page_title="Agentic CRAG Assistant", page_icon="🤖", layout="wide")

    # Session State Setup
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Agentic Corrective RAG (CRAG) Assistant 🤖")
    st.caption("Hybrid Search (BM25 + Vectors) ➔ Cross-Encoder Re-Ranking ➔ Corrective Web Fallback")

    # Sidebar Document Upload
    with st.sidebar:
        st.subheader("📁 Document Management")
        uploaded_pdfs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])

        if st.button("Build Knowledge Index", use_container_width=True):
            if not uploaded_pdfs:
                st.warning("Upload at least one PDF file first!")
            else:
                with st.spinner("Indexing documents, generating dense vectors & setting up re-ranker..."):
                    try:
                        raw_docs = extract_pdf_documents(uploaded_pdfs)
                        if not raw_docs:
                            st.error("Could not extract text from uploaded files.")
                            return

                        chunks = create_chunks(raw_docs)
                        st.session_state.pipeline = HybridCRAGPipeline(chunks)
                        st.session_state.messages = []
                        st.success(f"Indexed {len(chunks)} chunks across {len(uploaded_pdfs)} PDF(s)!")
                    except Exception as err:
                        st.error(f"Initialization error: {str(err)}")

    # Chat Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about your documents:"):
        if not st.session_state.pipeline:
            st.info("Please upload PDFs and build the knowledge index first.")
            return

        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving, evaluating & generating response..."):
                try:
                    result = st.session_state.pipeline.run(user_prompt)
                    response_text = result["generation"]

                    # Accurate status banner
                    if result.get("web_used"):
                        st.info("ℹ️ Local PDF context was insufficient. Answer supplemented using graded Tavily Web Search.")
                    elif result.get("web_search_needed"):
                        st.warning("⚠️ Local context was insufficient and web also returned no relevant results.")

                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")

if __name__ == "__main__":
    main()
    