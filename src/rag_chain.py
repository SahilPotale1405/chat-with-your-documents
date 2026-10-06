from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """Load the Groq-hosted LLM."""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


def get_retriever(vector_store, k=12):
    """Turn a vector store into a retriever returning the top-k most similar chunks."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_prompt():
    """Define how context + question get formatted for the LLM."""
    template = """You are a helpful assistant answering questions based on the provided document context.
Use ONLY the context below to answer the question. If the answer isn't in the context, say you don't know — do not make up information.

Context:
{context}

Question:
{question}

Answer:"""
    return ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Combine retrieved chunks into one text block, tagged with page numbers."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(vector_store):
    """Build the full retrieval-augmented generation chain."""
    retriever = get_retriever(vector_store)
    prompt = get_prompt()
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain