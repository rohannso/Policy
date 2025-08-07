import os
import streamlit as st
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .user-message {
        background: #e3f2fd;
        border-radius: 15px 15px 5px 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background: #f3e5f5;
        border-radius: 15px 15px 15px 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
    
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

# --- Helper Functions ---
@st.cache_resource
def get_embeddings_model():
    """Initialize and cache the embeddings model."""
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

def process_pdf(uploaded_file):
    """Process the uploaded PDF and create RAG chain."""
    if not os.getenv("GROQ_API_KEY"):
        st.error("🔑 GROQ_API_KEY not found. Please set it in your .env file.")
        return False
    
    embed_model = get_embeddings_model()
    if not embed_model:
        st.error("❌ Embedding model could not be initialized.")
        return False
    
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded file temporarily
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        
        status_text.text("📖 Loading PDF...")
        progress_bar.progress(20)
        
        # Load and process document
        loader = PyMuPDFLoader(f"temp_{uploaded_file.name}")
        docs = loader.load()
        
        status_text.text("✂️ Splitting document...")
        progress_bar.progress(40)
        
        # Setup document splitting
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        
        status_text.text("🗄️ Setting up vector store...")
        progress_bar.progress(60)
        
        # Setup vector store
        client = QdrantClient(":memory:")
        collection_name = "pdf_chat_collection"
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embed_model,
        )
        
        status_text.text("📚 Indexing document chunks...")
        progress_bar.progress(80)
        
        # Setup retriever
        docstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(docs, ids=None)
        
        status_text.text("🤖 Initializing AI model...")
        progress_bar.progress(95)
        
        # Setup RAG chain
        llm = ChatGroq(model_name="llama3-8b-8192")
        
        prompt_template = """
        You are a helpful AI assistant. Your task is to answer questions based *only* on the provided context from a document.
        Your answers should be clear, concise, and directly reference the information in the context.
        If the information is not in the context, you must clearly state: "The document does not contain information on this topic."
        Do not add any information that is not from the provided text.

        Context:
        {context}

        Question:
        {input}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Update session state
        st.session_state.rag_chain = rag_chain
        st.session_state.pdf_processed = True
        st.session_state.current_pdf = uploaded_file.name
        
        progress_bar.progress(100)
        status_text.text("✅ PDF processed successfully!")
        
        # Cleanup temp file
        os.remove(f"temp_{uploaded_file.name}")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        # Cleanup temp file if it exists
        if os.path.exists(f"temp_{uploaded_file.name}"):
            os.remove(f"temp_{uploaded_file.name}")
        return False

def get_response(query):
    """Get response from RAG chain."""
    if not st.session_state.rag_chain:
        return "Please upload and process a PDF first."
    
    try:
        response = st.session_state.rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"An error occurred: {e}"

# --- Main Interface ---
# Header
st.markdown("""
<div class="main-header">
    <h1>📚 AI-Powered PDF Chat Assistant</h1>
    <p>Upload your PDF document and ask questions about its content using advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for PDF upload and controls
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to start chatting"
    )
    
    if uploaded_file is not None:
        st.write(f"**Selected file:** {uploaded_file.name}")
        
        # Process button
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            if process_pdf(uploaded_file):
                st.success("✅ PDF processed successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to process PDF")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Reset all button
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.session_state.current_pdf = None
        st.rerun()
    
    st.divider()
    
    # Status information
    st.subheader("📊 Status")
    if st.session_state.pdf_processed:
        st.success(f"✅ Ready - {st.session_state.current_pdf}")
    else:
        st.warning("⏳ Upload a PDF to begin")
    
    st.info(f"💬 Chat messages: {len(st.session_state.chat_history)}")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="user-message">
                <strong>🧑‍💼 You:</strong><br>
                {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 Assistant:</strong><br>
                {bot_msg}
            </div>
            """, unsafe_allow_html=True)
    
    # Input area
    st.divider()
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your document:",
            placeholder="Type your question here...",
            key="user_input"
        )
        
        col_submit, col_space = st.columns([1, 3])
        with col_submit:
            submit_button = st.form_submit_button("Send 📤", type="primary")
    
    # Handle user input
    if submit_button and user_input:
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first!")
        else:
            # Add user message to history
            with st.spinner("🤔 Thinking..."):
                bot_response = get_response(user_input)
            
            st.session_state.chat_history.append((user_input, bot_response))
            st.rerun()

with col2:
    st.header("ℹ️ Instructions")
    
    st.markdown("""
    **How to use:**
    
    1. 📁 Upload a PDF file using the sidebar
    2. 🚀 Click "Process PDF" to analyze the document
    3. 💬 Ask questions about the document content
    4. 🔄 Use "Reset All" to start over
    
    **Tips:**
    - Ask specific questions about the document
    - The AI will only answer based on the PDF content
    - Use "Clear Chat" to reset conversation history
    """)
    
    st.info("🔑 Make sure you have GROQ_API_KEY in your .env file")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Powered by Streamlit, LangChain, and Groq AI
</div>
""", unsafe_allow_html=True)

# Check for API key on startup
if not os.getenv("GROQ_API_KEY"):
    st.error("""
    🔑 **GROQ_API_KEY not found!**
    
    Please create a `.env` file in your project directory with:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```
    
    Get your API key from: https://console.groq.com/keys
    """)