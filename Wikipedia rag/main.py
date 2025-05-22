import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# Load environment variables from .env file
load_dotenv()

# Ensure the Gemini API key is available
if not os.getenv("GEMINI_API_KEY"):
    st.error("Gemini API key not found. Please set it in the .env file as GEMINI_API_KEY.")
    st.stop()

INDEX_DIR = 'wiki-rag'
DEFAULT_PAGES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep Learning",
    "Neural network",
    "Convolutional neural network",
    "Reinforcement learning",
    "Supervised learning"
]

@st.cache_resource
def get_index(selected_pages):
    # Create a unique cache key based on selected pages
    cache_key = "_".join(sorted(selected_pages))
    index_dir = f"{INDEX_DIR}/{cache_key}"
    
    try:
        if os.path.isdir(index_dir):
            storage = StorageContext.from_defaults(persist_dir=index_dir)
            return load_index_from_storage(storage)
    except Exception as e:
        st.warning(f"Failed to load existing index: {e}. Creating a new index...")

    try:
        docs = WikipediaReader().load_data(pages=selected_pages, auto_suggest=False)
        if not docs:
            st.error("No documents found for the specified Wikipedia pages.")
            return None
        # Use a local embedding model
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
        index.storage_context.persist(persist_dir=index_dir)
        return index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        return None

@st.cache_resource
def get_query_engine(selected_pages):
    index = get_index(selected_pages)
    if index is None:
        return None
    # Use Gemini API for the LLM
    llm = Gemini(model_name="models/gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0.7)
    return index.as_query_engine(llm=llm, similarity_top_k=1)

def main():
    st.title('Wikipedia RAG Application')
    
    # Allow users to select pages to reduce initial load
    st.subheader("Select Wikipedia Pages to Query")
    selected_pages = st.multiselect(
        "Choose pages (fewer pages = faster response)",
        options=DEFAULT_PAGES,
        default=["Machine learning"]
    )
    
    if not selected_pages:
        st.warning("Please select at least one Wikipedia page.")
        return
    
    # Initialize query engine with selected pages
    with st.spinner("Indexing selected pages..."):
        qa = get_query_engine(selected_pages)
        if qa is None:
            st.error("Failed to initialize the query engine.")
            return

    # Query input
    question = st.text_input('Ask a question', placeholder="e.g., What is machine learning?")
    
    if st.button('Submit'):
        if not question:
            st.warning("Please enter a question.")
            return
        
        with st.spinner('Thinking...'):
            try:
                response = qa.query(question)
                st.subheader('Answer')
                st.write(response.response)
                
                st.subheader('Retrieved Context')
                for src in response.source_nodes:
                    st.text(src.node.get_content())
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == '__main__':
    main()