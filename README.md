This repository holds the RAG web application for Eliana's senior project


# RAG Web App with Streamlit, Ollama, and Chroma: Markdown Notes

## Overview
This guide walks you through creating a Retrieval-Augmented Generation (RAG) web app using Streamlit, DeepSeek-R1 (via Ollama), and Chroma as a vector database. Users can upload a PDF, ask questions, and get answers based on the document.

---

## Prerequisites
- **Python**: 3.9 or higher installed.
- **Ollama**: Installed from [ollama.ai](https://ollama.ai) to run DeepSeek-R1 locally.
- **Terminal Access**: For running Ollama and Streamlit commands.

---

## Step 1: Set Up Your Environment

### Terminal Setup
1. **Install Ollama**:
   - Download and install from [ollama.ai](https://ollama.ai).
2. **Pull DeepSeek-R1**:
   ```bash
   ollama pull deepseek-r1:7b
   ```
   - Downloads the model (run once).
3. **Run Ollama**:
   ```bash
   ollama run deepseek-r1
   ```
   - Keep this running in the background.

### Python Environment
1. **Create a Virtual Environment** (optional):
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install streamlit langchain sentence-transformers PyPDF2 chromadb ollama
   ```

---

## Step 2: Write the App Code

Create `RAGapp.py`:

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client

# Initialize clients
ollama_client = Client(host='http://localhost:11434')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Process uploaded PDF
def process_document(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Create Chroma vector store (with persistence)
def create_vector_store(chunks):
    client = chromadb.PersistentClient(path="rag_store")
    if "rag_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection("rag_collection")
    collection = client.create_collection("rag_collection")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return chunks, collection

# Retrieve relevant chunks
def retrieve_chunks(query, chunks, collection):
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results['documents'][0]

# Generate answer with DeepSeek-R1
def generate_response(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = ollama_client.chat(model='deepseek-r1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Streamlit UI
st.title("RAG Q&A Web App")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    if 'chunks' not in st.session_state or 'collection' not in st.session_state:
        with st.spinner("Processing document..."):
            chunks = process_document(uploaded_file)
            chunks, collection = create_vector_store(chunks)
            st.session_state['chunks'] = chunks
            st.session_state['collection'] = collection
        st.success("Document processed! You can now ask questions.")
    
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            context_chunks = retrieve_chunks(query, st.session_state['chunks'], st.session_state['collection'])
            answer = generate_response(query, context_chunks)
            st.write("**Answer:**", answer)
else:
    st.write("Please upload a document to start.")
```

---

## Step 3: Run the App
1. **Start Ollama**:
   ```bash
   ollama run deepseek-r1
   ```
2. **Launch the App**:
   ```bash
   streamlit run RAGapp.py
   ```
3. **Stop the App**:
   - In the terminal running Streamlit, press `Ctrl+C` (or `Cmd+C` on macOS).
   - This shuts down the server; the browser app stops working.
   - Persistent vector store (`rag_store`) remains on disk.

4. **Stop Ollama** (optional):
   - In the terminal running `ollama run deepseek-r1`, press `Ctrl+C`.
   - Closes the Ollama server.

### What Happens When You Stop It?
- **Streamlit Process**: The web server shuts down, and the Python script (`app.py`) stops executing.
- **Vector Database**:
  - If you’re using `chromadb.Client()` (in-memory), the vector store vanishes from RAM.
  - If you’re using `chromadb.PersistentClient(path="rag_store")` (like in the updated code), the vector store stays saved in the `rag_store` folder on disk, ready for the next run.
- **Ollama**: The Ollama server (running `ollama run deepseek-r1` in a separate terminal) keeps going unless you stop it separately (more on that below).

---

## Chroma Persistence Notes
- **In-Memory (`chromadb.Client()`)**:
  - Data lost when app stops.
- **Persistent (`chromadb.PersistentClient(path="rag_store")`)**:
  - Saved to `rag_store` folder, persists after stopping.
  - Overwrites old data on new uploads.

---

## Tips & Tweaks
- **Check Stop Worked**: Browser URL (`localhost:8501`) should stop responding after `Ctrl+C`.
- **Persistent Check**: Look for `rag_store` folder after stopping to confirm data saved.
- **Model Name**: Verify with `ollama list`.

---

## Troubleshooting
- **App Won’t Stop**: Ensure you’re in the right terminal; force-close with `Ctrl+Z` if stuck (Linux/macOS) or Task Manager (Windows).
- **Ollama Still Running**: Stop it separately if needed.
```

---

### Quick Recap
Just hit `Ctrl+C` in the Streamlit terminal to stop the app. With the persistent Chroma setup, your vector database will stick around in the `rag_store` folder, ready for the next time you fire it up. Let me know if you need anything else!