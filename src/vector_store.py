from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import uuid

def get_embedding_model():
    """Load a free, local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def create_vector_store(chunks, persist_directory=None):
    """Embed chunks and store them in a persistent, isolated ChromaDB collection."""
    if persist_directory is None:
        persist_directory = f"chroma_db/{uuid.uuid4().hex}"

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    return vector_store

def load_vector_store(persist_directory="chroma_db"):
    """Load an existing ChromaDB collection from disk."""
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    return vector_store