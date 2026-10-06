import streamlit as st
from src.document_processor import save_uploaded_file, load_and_chunk_pdf
from src.vector_store import create_vector_store
from src.rag_chain import build_rag_chain

st.set_page_config(page_title="Chat with Your Documents", page_icon="📄")
st.title("📄 Chat with Your Documents")
st.write("Upload a PDF and ask questions about its content.")

# Initialize session state variables (only runs once per session)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.session_state.vector_store is None:
        with st.spinner("Processing document... this may take a minute."):
            file_path = save_uploaded_file(uploaded_file)
            chunks = load_and_chunk_pdf(file_path)
            st.session_state.vector_store = create_vector_store(chunks)
            st.session_state.rag_chain = build_rag_chain(st.session_state.vector_store)
        st.success(f"Document processed! ({len(chunks)} chunks created)")

if st.session_state.rag_chain is not None:
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(question)
        st.write("### Answer")
        st.write(answer)