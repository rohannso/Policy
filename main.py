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

# --- Load Environment Variables ---
# Make sure to have a .env file with your GROQ_API_KEY
load_dotenv()

# --- Global Variables & Setup ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384 

# Initialize embeddings once to save time and resources.
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    print(f"Error initializing HuggingFace Embeddings: {e}")
    embeddings = None

def process_pdf(file):
    """
    Processes the uploaded PDF file once and creates a reusable RAG chain.
    This function is triggered only when a new file is uploaded.
    """
    if not file:
        return None, "Upload a PDF to begin."
    
    if not os.getenv("GROQ_API_KEY"):
        # This check is crucial for providing clear feedback to the user.
        gr.Warning("GROQ_API_KEY not found. Please set it in your .env file.")
        return None, "Error: GROQ API Key is not configured."
        
    if not embeddings:
        gr.Warning("Embedding model could not be initialized.")
        return None, "Error: Embedding model is not available."

    try:
        # --- 1. Load and Process the Document ---
        loader = PyMuPDFLoader(file.name)
        docs = loader.load()

        # --- 2. Setup Document Splitting ---
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000) # Larger parent chunks can sometimes provide better context
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

        # --- 3. Setup Vector Store (In-Memory Qdrant) ---
        client = QdrantClient(":memory:")
        collection_name = "pdf_collection"
        
        # Using try-except is a robust way to handle collection creation
        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Could not recreate collection, likely it doesn't exist. Creating new one. Error: {e}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

        # --- 4. Setup the Retriever ---
        docstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(docs, ids=None)

        # --- 5. Setup the RAG Chain with Groq ---
        llm = ChatGroq(model_name="llama3-8b-8192")

        prompt_template = """
        You are an expert analyst. Your task is to answer questions based *only* on the provided context from a document.
        If the information is not in the context, clearly state that the document does not contain the answer.
        Do not make up information.

        Context:
        {context}

        Question:
        {input}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        gr.Info("PDF processed successfully! You can now ask questions.")
        return rag_chain, "PDF is ready. Ask your question below."

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        gr.Warning("Failed to process the PDF. Please check the file and try again.")
        return None, f"Error processing PDF: {e}"

def answer_question(query, rag_chain):
    """
    Answers a user's question using the pre-processed RAG chain.
    """
    if rag_chain is None:
        gr.Warning("Please upload and process a PDF first.")
        return "Error: No document has been processed. Please upload a PDF."
    if not query:
        gr.Warning("Please enter a question.")
        return "Error: Please enter a question to get an answer."

    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        print(f"An error occurred during query invocation: {e}")
        return f"An unexpected error occurred: {e}"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="PDF Q&A with Groq") as demo:
    # State object to hold the RAG chain across interactions
    rag_chain_state = gr.State()

    gr.Markdown("# Efficient PDF Q&A with Groq & LangChain")
    gr.Markdown(
        "**Step 1:** Upload a PDF document. The system will process it and create a vector store.\n"
        "**Step 2:** Ask questions about the document's content."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_display = gr.Markdown("Status: Waiting for PDF...")
            
            gr.Markdown("---") # Separator
            
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Is eye surgery covered?",
                interactive=True
            )
            submit_button = gr.Button("Get Answer", variant="primary")
            
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Generated Answer")

    # When a file is uploaded, it triggers the processing function
    pdf_upload.upload(
        fn=process_pdf,
        inputs=[pdf_upload],
        outputs=[rag_chain_state, status_display]
    )

    # The submit button now uses the state to answer the question
    submit_button.click(
        fn=answer_question,
        inputs=[question_input, rag_chain_state],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("="*80)
        print("WARNING: GROQ_API_KEY environment variable not found.")
        print("Please create a '.env' file in the same directory as this script and add:")
        print('GROQ_API_KEY="gsk_..."')
        print("="*80)

    demo.launch(debug=True)
