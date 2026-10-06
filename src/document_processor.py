import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def save_uploaded_file(uploaded_file, save_dir="data"):
    """ Save a Streamlit- Uploaded file to disk and return its path."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path



def load_and_chunk_pdf(file_path, chunk_size=1000, chunk_overlap=150):
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        page.page_content = re.sub(r'\s+', ' ', page.page_content).strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    return chunks