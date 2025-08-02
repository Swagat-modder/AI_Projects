# 🤖 Wikipedia RAG Chatbot using Gemini & Streamlit

A smart chatbot that answers your questions using real Wikipedia content. Built with **Streamlit**, **LlamaIndex**, **HuggingFace Embeddings**, and the **Gemini LLM API**, this project demonstrates a simple yet powerful **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Features

- Select from popular Wikipedia topics
- Ask natural-language questions
- Retrieves relevant context using semantic search
- Generates answers using Gemini LLM
- Stores and reuses vector embeddings for speed

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace)
- **RAG Engine**: LlamaIndex
- **LLM**: Google Gemini (`models/gemini-1.5-flash`)
- **Storage**: Local directory (`wiki-rag/`)

---

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wikipedia-rag-chatbot.git
   cd wikipedia-rag-chatbot
