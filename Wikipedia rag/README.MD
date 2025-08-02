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

2. **Install dependencies
   ```bash
   pip install -r requirements.txt

3. **Set up environment variables

   GEMINI_API_KEY=your_gemini_api_key_here
4. **Run the app
   ```bash
   streamlit run main.py
## 📂 Folder Structure

   main.py               # Main app logic
   .env                  # API key file
   requirements.txt      # Dependencies
   wiki-rag/             # Auto-saved vector indices
## 💡Example Queries
   What is machine learning?
   How does a convolutional neural network work?
   What is reinforcement learning?

## 🙌 Acknowledgements
   LlamaIndex
   HuggingFace Sentence Transformers
   Gemini API

