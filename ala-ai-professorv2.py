import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient
import hashlib
from streamlit_pdf_viewer import pdf_viewer
import tempfile 
import os

# Initialize API keys
# Option 1: Using environment variables
google_api_key = os.getenv('GOOGLE_API_KEY')
tvly_api_key = os.getenv('TAVILY_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Option 2: Using Streamlit secrets (uncomment if using secrets.toml)
# if 'google_api_key' in st.secrets:
#     google_api_key = st.secrets['google_api_key']
#     tvly_api_key = st.secrets['tvly_api_key']
#     openai_api_key = st.secrets['openai_api_key']

# Validate API keys
if not all([google_api_key, tvly_api_key, openai_api_key]):
    st.error("Please set up your API keys in environment variables or secrets.toml")
    st.stop()

# Initialize Tavily client
web_tool_search = TavilyClient(api_key=tvly_api_key)

# Set up Streamlit page
st.set_page_config(page_title="AI Professor", page_icon="👨‍🏫")
st.title("👨‍🏫 AI Professor")

def get_pdf_text(pdf_docs):
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_response(user_query, chat_history, vector_store):
    if vector_store is None:
        return "Please upload a PDF document first."
    
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the document provided:

    Context: {context}
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    try:
        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=openai_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=1,
            max_tokens=1024
        )
        
        docs = vector_store.similarity_search(user_query)
        context = "\n".join(doc.page_content for doc in docs)

        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "user_question": user_query,
        })
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_youtube_url(query):
    try:
        response = web_tool_search.search(
            query=query,
            search_depth="basic",
            include_domains=["youtube.com"],
            max_results=1
        )
        
        for result in response['results']:
            if 'youtube.com/watch' in result['url']:
                return result['url']
        
        return None
    except Exception as e:
        st.error(f"Error searching for video: {str(e)}")
        return None

def get_pdfs_hash(pdf_docs):
    combined_hash = hashlib.md5()
    if isinstance(pdf_docs, list):
        for pdf in pdf_docs:
            content = pdf.read()
            combined_hash.update(content)
            pdf.seek(0)
    else:
        content = pdf_docs.read()
        combined_hash.update(content)
        pdf_docs.seek(0)
    return combined_hash.hexdigest()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Chatbot professor assistant. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_pdfs_hash" not in st.session_state:
    st.session_state.current_pdfs_hash = None

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Chat input
user_query = st.chat_input("Type your message here...")

# Sidebar
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=False, key="pdf_uploader")
    quiz_button = st.button("🗒️ Make a quiz", type="primary")
    video_button = st.button("📺 Search a video on the topic")
    view = st.toggle("👁️ View PDF")
    
    if view and pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_docs.read())
            temp_pdf_path = temp_file.name
        pdf_viewer(temp_pdf_path, width=800)
        
        # Custom CSS for sidebar
        st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                width: 600px;
                min-width: 600px;
                max-width: 800px;
                background-color: #f0f2f6;
            }
            .css-1lcbmhc {
                margin-left: 360px;
                padding: 1rem;
            }
            .block-container {
                max-width: 800px;
                min-width: 600px;
                margin: auto;
            }
            .stChatMessage {
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
            }
        </style>
        """, unsafe_allow_html=True)

# Process PDF upload
if pdf_docs:
    new_hash = get_pdfs_hash(pdf_docs)
    if new_hash != st.session_state.current_pdfs_hash:
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        st.session_state.vector_store = get_vector_store(text_chunks)
        st.session_state.current_pdfs_hash = new_hash
        st.success("The document has been updated!")

# Handle user query
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query, unsafe_allow_html=True)
    
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = get_response(user_query, st.session_state.chat_history, st.session_state.vector_store)
            st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))

# Show message if no PDF is uploaded
if pdf_docs is None:
    st.write("Please upload your PDF course before starting the chat.")

# Handle quiz generation
if quiz_button:
    with st.spinner("Generating quiz..."):
        quiz_prompt = """
        Based on the document content, create a quiz with 5 multiple choice questions.
        For each question:
        1. Ask a clear, specific question
        2. Provide 4 options labeled A, B, C, D
        3. Make sure the options are plausible but distinct
        4. Don't reveal the correct answer

        Format each question like this:
        Question X:
        **A)**
        **B)**
        **C)**
        **D)**
        """
        with st.chat_message("AI"):
            response = get_response(quiz_prompt, st.session_state.chat_history, st.session_state.vector_store)
            st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))

# Handle video search
if video_button:
    with st.spinner("Searching for relevant video..."):
        video_prompt = """
        Extract the main topic and key concepts from the document or from the last conversation in 3-4 words maximum.
        Focus on the core subject matter only.
        Do not include any additional text or explanation.
        Example format: "machine learning neural networks" or "quantum computing basics"
        """
        with st.chat_message("AI"):
            response = get_response(video_prompt, st.session_state.chat_history, st.session_state.vector_store)
            youtube_url = get_youtube_url(f"Course on {response}")
            if youtube_url:
                st.write(f"📺 Here's a video about {response}:")
                st.video(youtube_url)
                video_message = f"📺 Here's a video about {response}:\n{youtube_url}"
                st.session_state.chat_history.append(AIMessage(content=video_message))
