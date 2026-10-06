"""
Quick manual test script to verify the RAG pipeline works end-to-end,
independent of the Streamlit UI. Run with: python test_pipeline.py
"""

from src.document_processor import load_and_chunk_pdf
from src.vector_store import create_vector_store

# Change this to the path of one of your own PDFs
pdf_path = "data/She Was Only Mine.pdf"

chunks = load_and_chunk_pdf(pdf_path)
print(f"Number of chunks created: {len(chunks)}")
print(f"First chunk preview:\n{chunks[0].page_content[:300]}")

vector_store = create_vector_store(chunks)
print("Vector store created successfully!")

from src.rag_chain import build_rag_chain

rag_chain = build_rag_chain(vector_store)

question = "What is this document about?"
answer = rag_chain.invoke(question)
print(f"\nQuestion: {question}")
print(f"Answer: {answer}")


# Debug: print a sample of pages/chunks to see what's actually in the PDF
for i in [0, 100, 200, 300, 400]:
    print(f"\n--- Chunk {i} ---")
    print(chunks[i].page_content[:200])
    print(f"(from page {chunks[i].metadata.get('page')})")