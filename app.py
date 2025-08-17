from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import os
import time
import shutil

# === Load .env if needed ===
load_dotenv()

# === Streamlit Page Settings ===
st.set_page_config(page_title=" RAG", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #000;
            color: white;
        }
        .user-msg {
            background-color: #1a1a1a;
            padding: 12px;
            border-radius: 10px;
            text-align: right;
            margin: 5px 0;
        }
        .bot-msg {
            background-color: #262626;
            padding: 12px;
            border-radius: 10px;
            text-align: left;
            margin: 5px 0;
        }
        .timestamp {
            font-size: 0.7em;
            color: gray;
        }
        .chat-container {
            max-height: 60vh;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.title(" ,RAG Chatbot with Memory & File Upload")

# === Sidebar ===
temperature = st.sidebar.slider("🎛️ Temperature", 0.0, 1.0, 0.7)
uploaded_file = st.sidebar.file_uploader("📄 Upload PDF File", type=["pdf"])
reset_index = st.sidebar.button("🔁 Reset Vector Store")

# === Session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("system", "You are a helpful AI assistant.")]
if "formatted_history" not in st.session_state:
    st.session_state.formatted_history = []
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# === Paths ===
VECTOR_DIR = "faiss_index"
PDF_DIR = "uploaded_docs"

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# === Reset Vector Store ===
if reset_index:
    shutil.rmtree(VECTOR_DIR)
    shutil.rmtree(PDF_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    st.success("✅ Vector store and documents reset.")

# === Auto Create Vector Store ===
def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)

if uploaded_file:
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    create_vector_store(file_path)
    st.success(f"✅ Vector store updated with {uploaded_file.name}")

# === Load LLM and Vector Store ===
llm = Ollama(model="llama3", temperature=temperature)
output_parser = StrOutputParser()

def load_vector_store():
    embeddings = OllamaEmbeddings(model="llama3")
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    return None

retriever = None
vectorstore = load_vector_store()
if vectorstore:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# === Chat UI ===
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg, ts in st.session_state.formatted_history:
    html = f"""
        <div class="{ 'user-msg' if role == 'user' else 'bot-msg' }">
            <div>{msg}</div>
            <div class="timestamp">{ts}</div>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# === Clear Chat Confirmation ===
if st.button("🗑️ Clear Chat"):
    st.session_state.confirm_clear = True

if st.session_state.confirm_clear:
    st.warning("⚠️ Are you sure you want to clear the chat?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, delete"):
            st.session_state.chat_history = [("system", "You are a helpful AI assistant.")]
            st.session_state.formatted_history = []
            st.session_state.confirm_clear = False
            st.rerun()
    with col2:
        if st.button("❌ No, cancel"):
            st.session_state.confirm_clear = False

# === Chat Form ===
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Ask Anything", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.formatted_history.append(("user", user_input, timestamp))

    if retriever:
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_input, "chat_history": []})
            response = result["answer"]
            sources = "\n\n**Sources:**\n" + "\n".join([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]])
            response += sources
    else:
        prompt = ChatPromptTemplate.from_messages(st.session_state.chat_history)
        chain = prompt | llm | output_parser
        with st.spinner("Thinking..."):
            response = chain.invoke({})

    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("assistant", response))
    st.session_state.formatted_history.append(("assistant", response, timestamp))
    st.rerun()

# === Download Chat ===
if st.download_button("⬇️ Download Chat", "\n\n".join(f"{r.upper()} [{t}]: {m}" for r, m, t in st.session_state.formatted_history), file_name="chat_history.txt"):
    st.success("📁 Chat downloaded!")
