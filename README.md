# 📄 Chat with Your Documents — RAG-based PDF Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask natural-language questions about its content. Built as a hands-on project to learn and demonstrate practical LLM application development using LangChain.

🔗 **Live Demo:** [chat-with-your-documents1.streamlit.app](https://chat-with-your-documents1.streamlit.app)

📂 **Source Code:** [github.com/SahilPotale1405/chat-with-your-documents](https://github.com/SahilPotale1405/chat-with-your-documents)

---

## 🧠 What It Does

Upload any PDF (notes, reports, research papers) and ask questions about it in plain English. The app retrieves the most relevant sections of the document and uses an LLM to generate accurate, grounded answers — rather than relying on the LLM's own (potentially outdated or hallucinated) knowledge.

**Example use case:** Upload a technical project report and ask "What is the tech stack used?" or "What are the limitations mentioned?" — get accurate answers sourced directly from the document.

---

## 🏗️ How It Works

1. **Document Loading & Chunking** — The PDF is parsed page-by-page and split into overlapping text chunks using `RecursiveCharacterTextSplitter`, preserving semantic coherence.
2. **Embedding** — Each chunk is converted into a vector using a free, local HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`) — no API costs, runs on CPU.
3. **Vector Storage** — Embeddings are stored in **ChromaDB**, a persistent vector database, enabling fast similarity search.
4. **Retrieval** — When a question is asked, it's embedded the same way, and the most semantically similar chunks are retrieved from ChromaDB.
5. **Generation** — Retrieved chunks + the question are passed to a prompt template and sent to a hosted LLM (**Groq's `openai/gpt-oss-20b`**) via LangChain's LCEL chains, generating a grounded, context-aware answer.

```
PDF Upload → Chunking → Embedding (HuggingFace) → ChromaDB Storage
                                                          ↓
User Question → Embedding → Similarity Search → Top-K Relevant Chunks
                                                          ↓
                                    Chunks + Question → Prompt → LLM (Groq) → Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | ChromaDB |
| LLM | Groq-hosted `openai/gpt-oss-20b` (free tier) |
| UI | Streamlit |
| PDF Parsing | PyPDF |
| Deployment | Streamlit Community Cloud |
| Package Management | uv |

---

## 📸 Screenshots

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/4f19182e-e9f9-4b20-9e4b-571cfc29c41e" />

---

## 🚀 Running Locally

1. Clone the repo:
```bash
git clone https://github.com/SahilPotale1405/chat-with-your-documents.git
cd chat-with-your-documents
```

2. Install dependencies (using [uv](https://github.com/astral-sh/uv)):
```bash
uv sync
```

3. Get a free Groq API key from [console.groq.com](https://console.groq.com), then create a `.env` file in the project root with:
```
GROQ_API_KEY=your_key_here
```

4. Run the app:
```bash
streamlit run app.py
```

---

## 💡 What I Learned / Design Decisions

- **Why HuggingFace embeddings instead of OpenAI's?** Keeps the project fully free and runs locally — no per-request API costs for the embedding step, which is called far more often than the LLM.
- **Why Groq for the LLM?** Free tier with genuinely fast inference, important for a responsive live demo.
- **Low temperature (0.2) for generation:** RAG prioritizes factual grounding over creativity — lower temperature reduces hallucination risk.
- **Chunk size/overlap tuning (1000 chars, 150 overlap):** Balances context preservation against retrieval precision.
- **Known limitation:** Retrieval is based on semantic similarity, not document structure — ambiguous or overlapping sections (e.g. two differently-scoped "Limitations" sections in one document) can occasionally cause the wrong-but-related chunk to be retrieved. Mitigated here by retrieving a wider top-k, but a production system might add section-aware metadata filtering.
- **Per-upload isolated vector stores:** Each processed document gets its own UUID-based ChromaDB directory, preventing data from different uploads from mixing.

---

## 🔮 Future Improvements

- Support multiple PDFs in a single session
- Source citations (show which page an answer came from)
- Section/metadata-aware retrieval for better precision on ambiguous queries
- Persistent chat history

---

## 📬 Contact

**Sahil Potale** — [LinkedIn](https://www.linkedin.com/in/sahil-potale-725680229) — [Email](mailto:sahilpotale2004@gmail.com)
