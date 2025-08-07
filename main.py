import os
import gradio as gr
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
import time

# --- Load Environment Variables ---
# Make sure to have a .env file with your GROQ_API_KEY
load_dotenv()

# --- Global Variables & Setup ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384 

# Initialize embeddings once to save time and resources.
# Using a global variable to avoid re-initializing the model.
embeddings = None
def get_embeddings_model():
    """Initializes and returns the HuggingFace embeddings model."""
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error initializing HuggingFace Embeddings: {e}")
            # In a real application, you might want to handle this more gracefully.
            # For this Gradio app, we'll rely on the UI to inform the user.
    return embeddings

def process_pdf(file, progress=gr.Progress(track_tqdm=True)):
    """
    Processes the uploaded PDF file, creates a RAG chain, and returns an initial chat message.
    This function is triggered only when a new file is uploaded.
    """
    if not file:
        return None, [(None, "Please upload a PDF to begin.")]

    if not os.getenv("GROQ_API_KEY"):
        gr.Warning("GROQ_API_KEY not found. Please set it in your .env file.")
        return None, [(None, "Error: GROQ API Key is not configured. Please add it to your .env file and restart.")]

    embed_model = get_embeddings_model()
    if not embed_model:
        gr.Warning("Embedding model could not be initialized.")
        return None, [(None, "Error: Embedding model is not available. Check your internet connection or model configuration.")]

    try:
        progress(0, desc="Starting PDF Processing...")
        
        # --- 1. Load and Process the Document ---
        progress(0.1, desc="Loading PDF...")
        loader = PyMuPDFLoader(file.name)
        docs = loader.load()

        # --- 2. Setup Document Splitting ---
        progress(0.3, desc="Splitting document into chunks...")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

        # --- 3. Setup Vector Store (In-Memory Qdrant) ---
        progress(0.5, desc="Setting up vector store...")
        client = QdrantClient(":memory:")
        collection_name = "pdf_chat_collection"
        
        # Use recreate_collection for simplicity, it handles both creation and reset.
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embed_model,
        )

        # --- 4. Setup the Retriever ---
        progress(0.7, desc="Indexing document chunks...")
        docstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(docs, ids=None)

        # --- 5. Setup the RAG Chain with Groq ---
        progress(0.9, desc="Initializing AI model...")
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
        
        gr.Info("PDF processed successfully!")
        initial_chat = [(None, f"I've finished processing **{os.path.basename(file.name)}**. You can now ask me questions about it.")]
        return rag_chain, initial_chat

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        gr.Warning("Failed to process the PDF. Please check the file format and try again.")
        error_chat = [(None, f"Sorry, I ran into an error while processing the PDF: {e}")]
        return None, error_chat

def add_text(history, text):
    """Adds the user's question to the chat history."""
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def chat_flow(rag_chain, history):
    """
    Handles the main chat interaction, from getting the user query to streaming the bot's answer.
    """
    if rag_chain is None:
        gr.Warning("Please upload and process a PDF first.")
        history[-1][1] = "I can't answer without a document. Please upload a PDF."
        yield history
        return

    user_query = history[-1][0]
    if not user_query:
        gr.Warning("Please enter a question.")
        history[-1][1] = "It looks like you didn't ask a question. Please type one in the box below."
        yield history
        return
    
    try:
        # Stream the response for a better user experience
        response = rag_chain.invoke({"input": user_query})
        answer = response["answer"]
        
        history[-1][1] = ""
        for char in answer:
            history[-1][1] += char
            time.sleep(0.01) # Small delay for streaming effect
            yield history
            
    except Exception as e:
        print(f"An error occurred during query invocation: {e}")
        history[-1][1] = f"An unexpected error occurred. Please try again. Details: {e}"
        yield history

def clear_all():
    """Clears all components of the interface."""
    return None, None, None

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), title="PDF Chat") as demo:
    # State object to hold the RAG chain across interactions
    rag_chain_state = gr.State()

    gr.Markdown("# Chat with your PDF using Groq & LangChain")
    gr.Markdown("Upload a PDF, wait for it to be processed, and then ask questions in the chat window below.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)
                clear_button = gr.Button("Clear & Start Over", variant="stop", scale=0)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat Window",
                bubble_full_width=False,
                avatar_images=(None, "https://i.imgur.com/9kAC4pQ.png"), # User, Bot
                height=500
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question", 
                    placeholder="Ask something about the document...",
                    scale=4,
                    autofocus=True
                )
                submit_button = gr.Button("Send", variant="primary", scale=1, min_width=10)

    # --- Event Handlers ---

    # 1. When a file is uploaded, process it.
    pdf_upload.upload(
        fn=process_pdf,
        inputs=[pdf_upload],
        outputs=[rag_chain_state, chatbot],
        show_progress="full"
    )

    # 2. When the user submits their question (via button or enter).
    chat_msg = question_input.submit(
        fn=add_text,
        inputs=[chatbot, question_input],
        outputs=[chatbot, question_input],
        queue=False,
    ).then(
        fn=chat_flow,
        inputs=[rag_chain_state, chatbot],
        outputs=[chatbot]
    )
    chat_msg.then(lambda: gr.update(interactive=True), None, [question_input], queue=False)


    submit_button.click(
        fn=add_text,
        inputs=[chatbot, question_input],
        outputs=[chatbot, question_input],
        queue=False,
    ).then(
        fn=chat_flow,
        inputs=[rag_chain_state, chatbot],
        outputs=[chatbot]
    )
    submit_button.click(lambda: gr.update(interactive=True), None, [question_input], queue=False)

    # 3. When the clear button is clicked.
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[rag_chain_state, chatbot, pdf_upload],
        queue=False
    )

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("="*80)
        print("WARNING: GROQ_API_KEY environment variable not found.")
        print("Please create a '.env' file in the same directory as this script and add:")
        print('GROQ_API_KEY="your_groq_api_key_here"')
        print("You can get a key from https://console.groq.com/keys")
        print("="*80)

    demo.queue().launch(debug=True)
