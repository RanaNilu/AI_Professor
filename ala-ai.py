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

google_api_key = st.secrets['google_api_key']
tvly_api_key = st.secrets['tvly_api_key']
openai_api_key = st.secrets['openai_api_key']

web_tool_search = TavilyClient(api_key= tvly_api_key)

st.set_page_config(page_title="AI Professor", page_icon="👨‍🏫" )
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_response(user_query, chat_history, vector_store):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the document provided:

    Context: {context}
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
    base_url = "https://api.groq.com/openai/v1",
    openai_api_key = openai_api_key,
    model_name = "llama-3.1-8b-instant",
    temperature=1,
    max_tokens=1024
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    docs = vector_store.similarity_search(user_query)

    context = "\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "user_question": user_query,
    })

def get_youtube_url(query):
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Chatbot professor assistant. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_pdfs_hash" not in st.session_state:
    st.session_state.current_pdfs_hash = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message here...")

with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files ", accept_multiple_files=False, key="pdf_uploader")
        quizz_button= st.button("🗒️ Make a quizz", type="primary")
        video_button = st.button("📺 Search a video on the topic")
        view = st.toggle("👁️ View PDF")
        if view and pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_docs.read())
                temp_pdf_path = temp_file.name  

            pdf_viewer(temp_pdf_path, width=800)
            st.markdown(    
            """
    <style>
        /* Restrict sidebar width and ensure it doesn't cover main content */
        section[data-testid="stSidebar"] {
            width: 600px; /* Adjust this to control sidebar width */
            min-width: 600px;
            max-width: 800px;
            background-color: #f0f2f6;
        }

        /* Make main container responsive to sidebar width */
        .css-1lcbmhc { /* Main container class */
            margin-left: 360px; /* Slightly larger than sidebar width */
            padding: 1rem;
        }

        /* Control main content width for responsive behavior */
        .block-container {
            max-width: 800px; /* Prevent overlap */
            min-width: 600px;
            margin: auto;
        }

        /* Styling adjustments for chat messages */
        .stChatMessage {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True
    )     
            
    
if pdf_docs:
        new_hash = get_pdfs_hash(pdf_docs)
        if new_hash != st.session_state.current_pdfs_hash:
            text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.session_state.current_pdfs_hash = new_hash
            st.success("The document has been updated!")

if user_query is not None and user_query != "": 
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query,unsafe_allow_html=True)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = (get_response(user_query, st.session_state.chat_history, st.session_state.vector_store))
            st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))

if pdf_docs is None :
    st.write("Please upload your PDF course before starting the chat.")

if quizz_button :
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


if video_button :
     with st.spinner("Searching for relevant video..."): 
        video_prompt = """
        Extract the main topic and key concepts from the document or from the last conversation in 3-4 words maximum.
                Focus on the core subject matter only.
                Do not include any additional text or explanation.
                Example format: "machine learning neural networks" or "quantum computing basics"
        """
        with  st.chat_message("AI"):
            response = get_response(video_prompt, st.session_state.chat_history, st.session_state.vector_store)
            youtube_url = get_youtube_url(f"Course on {response}")
            if youtube_url:
                            st.write(f"📺 Here's a video about {response}:")
                            st.video(youtube_url)
                        
                            video_message = f"📺 Here's a video about {response}:\n{youtube_url}"
                            st.session_state.chat_history.append(AIMessage(content=video_message))
                            