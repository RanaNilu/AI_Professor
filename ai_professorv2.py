import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import hashlib
from streamlit_pdf_viewer import pdf_viewer
import tempfile
import os

# app.py
import streamlit as st
from api_keys import GOOGLE_API_KEY, TAVILY_API_KEY, OPENAI_API_KEY

# Use the API keys
st.write(f"Google API Key: {GOOGLE_API_KEY}")
st.write(f"Tavily API Key: {TAVILY_API_KEY}")
st.write(f"OpenAI API Key: {OPENAI_API_KEY}")

# Linux/Mac
export GOOGLE_API_KEY='AIzaSyDiZjRdBVZNqmhCQHnqDjz_fjgdfARyZp4'
export TAVILY_API_KEY='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'
export OPENAI_API_KEY='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'


# API Keys
google_api_key = st.secrets.get("google_api_key", "")
openai_api_key = st.secrets.get("openai_api_key", "")

# Streamlit Page Configuration
st.set_page_config(page_title="AI Professor", page_icon="👨‍🏫")
st.title("👨‍🏫 AI Professor")

# Helper Functions
def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    if isinstance(pdf_docs, list):
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    else:
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_response(user_query, vector_store):
    """Generate a response using LangChain."""
    template = """
    You are a helpful assistant. Use the context from the provided document to answer the user's question.

    Context: {context}
    User Question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.7)
    docs = vector_store.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    response = llm.predict(prompt.format(context=context, user_question=user_query))
    return response

def get_pdf_hash(pdf_docs):
    """Generate a hash for the uploaded PDFs."""
    combined_hash = hashlib.md5()
    content = pdf_docs.read()
    combined_hash.update(content)
    pdf_docs.seek(0)  # Reset the stream position
    return combined_hash.hexdigest()

# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None

# Sidebar for PDF Upload
with st.sidebar:
    st.title("Upload Course Material")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        view_pdf = st.checkbox("View PDF")
        if view_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.read())
                temp_pdf_path = temp_file.name
            pdf_viewer(temp_pdf_path, width=800)

# Process PDF and Generate Vector Store
if uploaded_pdf:
    new_hash = get_pdf_hash(uploaded_pdf)
    if new_hash != st.session_state.current_pdf_hash:
        pdf_text = get_pdf_text(uploaded_pdf)
        text_chunks = get_text_chunks(pdf_text)
        st.session_state.vector_store = get_vector_store(text_chunks)
        st.session_state.current_pdf_hash = new_hash
        st.success("PDF processed successfully! You can now ask questions.")

# Chat Input and Response
user_query = st.text_input("Ask your question:")
if user_query:
    if st.session_state.vector_store:
        with st.spinner("Thinking..."):
            response = get_response(user_query, st.session_state.vector_store)
            st.markdown(f"**AI Professor:** {response}")
    else:
        st.error("Please upload a PDF file first.")
