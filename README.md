<div align="center">

# 🔍 RAG Document Q&A System

**A production-ready Retrieval-Augmented Generation pipeline for intelligent document question answering**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-0.3+-1C3C3C?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-3DDC84?style=for-the-badge)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Evaluation](#-evaluation)

</div>

---

## 📌 Overview

This project implements a complete **RAG (Retrieval-Augmented Generation)** pipeline that allows users to upload documents (PDF, TXT, DOCX) and ask questions about their content using natural language. The system intelligently retrieves relevant document chunks and generates accurate, context-grounded answers with source citations.

### Why RAG?

Traditional LLMs hallucinate and can't access private data. RAG solves both problems by grounding LLM responses in your actual documents, providing accurate answers with verifiable sources.

---

## ✨ Features

- 📄 **Multi-format Document Ingestion** — PDF, TXT, DOCX support with intelligent chunking
- 🧠 **Semantic Search** — OpenAI embeddings + ChromaDB vector store for accurate retrieval
- 💬 **Conversational Q&A** — Multi-turn conversations with context memory
- 📎 **Source Citations** — Every answer includes references to source documents and pages
- 🔄 **Hybrid Search** — Combines semantic similarity with keyword matching (BM25)
- 📊 **RAG Evaluation** — Built-in metrics: Faithfulness, Answer Relevancy, Context Precision
- 🎨 **Streamlit UI** — Clean, intuitive web interface for document upload and chat
- ⚡ **Streaming Responses** — Real-time token streaming for better UX
- 🔧 **Configurable Pipeline** — Easy to swap LLMs, embeddings, and vector stores

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Doc Upload   │  │  Chat Interface  │  │  Source Viewer   │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────▲─────────┘  │
└─────────┼──────────────────┼──────────────────────┼─────────────┘
          │                  │                      │
          ▼                  ▼                      │
┌─────────────────┐  ┌──────────────┐              │
│  Document       │  │   Query      │              │
│  Processor      │  │   Engine     │              │
│  ┌───────────┐  │  │  ┌────────┐  │    ┌────────┴────────┐
│  │ Loader    │  │  │  │Retriev.│──┼───▶│ Response        │
│  │ Splitter  │  │  │  │Rerank  │  │    │ Generator       │
│  │ Embedder  │  │  │  │Filter  │  │    │ (with citations)│
│  └─────┬─────┘  │  │  └────────┘  │    └─────────────────┘
└────────┼────────┘  └──────┬───────┘
         │                  │
         ▼                  ▼
┌──────────────────────────────────┐
│         ChromaDB Vector Store    │
│    (Persistent / In-Memory)      │
└──────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/altafpinjari2001/rag-document-qa.git
cd rag-document-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Application

```bash
# Start Streamlit UI
streamlit run app.py

# Or run via CLI
python -m src.cli --document path/to/doc.pdf --query "What is this document about?"
```

---

## 💻 Usage

### 1. Upload Documents
Upload one or more documents through the Streamlit interface. Supported formats: PDF, TXT, DOCX.

### 2. Ask Questions
Type your question in the chat input. The system will:
1. Convert your question to an embedding
2. Retrieve the most relevant document chunks
3. Generate an answer grounded in the retrieved context
4. Display source citations with page numbers

### 3. Python API

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    chunk_size=1000,
    chunk_overlap=200,
)

# Ingest documents
rag.ingest("documents/research_paper.pdf")

# Query with sources
response = rag.query("What are the main findings?")
print(response.answer)
print(response.sources)
```

---

## 📊 Evaluation

Built-in RAG evaluation using custom metrics:

```bash
python -m src.evaluate --test-dataset data/eval_qa_pairs.json
```

| Metric | Score |
|--------|-------|
| Faithfulness | 0.92 |
| Answer Relevancy | 0.89 |
| Context Precision | 0.87 |
| Context Recall | 0.91 |

---

## 📁 Project Structure

```
rag-document-qa/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── document_processor.py # Document loading & chunking
│   ├── embeddings.py         # Embedding model wrapper
│   ├── vector_store.py       # ChromaDB operations
│   ├── retriever.py          # Hybrid retrieval logic
│   ├── generator.py          # LLM response generation
│   ├── pipeline.py           # End-to-end RAG pipeline
│   ├── evaluate.py           # RAG evaluation metrics
│   └── cli.py                # Command-line interface
├── tests/
│   ├── test_document_processor.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── data/
│   └── eval_qa_pairs.json    # Evaluation dataset
├── .github/
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline
├── LICENSE
└── .gitignore
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>⭐ Star this repo if you find it useful!</b>
</div>
