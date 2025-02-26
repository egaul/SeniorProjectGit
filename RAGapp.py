import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from ollama import Client

# Initialize Ollama client and embedder
ollama_client = Client(host='http://localhost:11434')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Process document (same as before)
def process_document(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Create Chroma vector store
def create_vector_store(chunks):
    client = chromadb.Client()  # In-memory client; use PersistentClient for disk storage
    collection = client.create_collection("rag_collection")
    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()  # Chroma needs list format
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return chunks, collection

# Retrieve with Chroma
def retrieve_chunks(query, chunks, collection):
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results['documents'][0]  # Chroma returns top matches

# Generate response (same as before)
def generate_response(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = ollama_client.chat(model='deepseek-r1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Streamlit UI (adapted)
st.title("RAG Q&A Web App with Chroma")
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