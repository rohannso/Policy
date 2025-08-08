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
# MODIFICATION: Updated CSS to support new chat elements
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
    
    /* Making chat messages look nicer */
    [data-testid="chat-message-container"] {
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User message styling */
    [data-testid="chat-message-container"]:has([data-testid="stChatMessageContent"] p) {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
    }

    /* Bot message styling */
    [data-testid="chat-message-container"]:has([data-testid="stChatMessageContent"] div[data-testid="stMarkdownContainer"]) {
        background: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }

    .st-emotion-cache-janbn0 {
      color: black;
    }

    .st-emotion-cache-4oy321 {
      color: black;
    }

</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
# MODIFICATION: Changed chat_history to the more standard 'messages' for chat apps
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
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
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        
        status_text.text("📖 Loading PDF...")
        progress_bar.progress(10)
        
        loader = PyMuPDFLoader(f"temp_{uploaded_file.name}")
        docs = loader.load()
        
        status_text.text("✂️ Splitting document into chunks...")
        progress_bar.progress(30)
        
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        
        status_text.text("🗄️ Setting up vector store...")
        progress_bar.progress(50)
        
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
        progress_bar.progress(70)
        
        docstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(docs, ids=None)
        
        status_text.text("🤖 Initializing AI model...")
        progress_bar.progress(90)
        
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
        
        st.session_state.rag_chain = rag_chain
        st.session_state.pdf_processed = True
        st.session_state.current_pdf = uploaded_file.name
        
        progress_bar.progress(100)
        status_text.text("✅ PDF processed successfully!")
        
        os.remove(f"temp_{uploaded_file.name}")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        if os.path.exists(f"temp_{uploaded_file.name}"):
            os.remove(f"temp_{uploaded_file.name}")
        return False

# MODIFICATION: Removed the synchronous get_response function. Streaming is handled directly in the UI.

# --- Main Interface ---
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
        
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            if process_pdf(uploaded_file):
                st.success("✅ PDF processed successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to process PDF")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.rag_chain = None
        st.session_state.messages = []
        st.session_state.pdf_processed = False
        st.session_state.current_pdf = None
        st.rerun()
    
    st.divider()
    
    st.subheader("📊 Status")
    if st.session_state.pdf_processed:
        st.success(f"✅ Ready - {st.session_state.current_pdf}")
    else:
        st.warning("⏳ Upload a PDF to begin")
    
    st.info(f"💬 Chat messages: {len(st.session_state.messages)}")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # MODIFICATION: Display existing chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # MODIFICATION: Use st.chat_input for a better, more modern chat UI
    if user_input := st.chat_input("Ask a question about your document..."):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first!")
        else:
            # Add user message to session state and display it
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and stream the bot's response
            with st.chat_message("assistant"):
                # This is the core of the streaming implementation
                def stream_generator():
                    """A generator function that yields chunks of the 'answer' from the RAG chain stream."""
                    try:
                        for chunk in st.session_state.rag_chain.stream({"input": user_input}):
                            if "answer" in chunk:
                                yield chunk["answer"]
                    except Exception as e:
                        yield f"An error occurred: {e}"

                # Use st.write_stream to display the streaming content
                response = st.write_stream(stream_generator)

            # Add the full bot response to the session state
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("ℹ️ Instructions")
    st.markdown("""
    **How to use:**
    
    1. 📁 Upload a PDF file using the sidebar
    2. 🚀 Click "Process PDF" to analyze the document
    3. 💬 Ask questions in the chat box at the bottom
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
