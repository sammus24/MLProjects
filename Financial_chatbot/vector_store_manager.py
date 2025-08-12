import os
import boto3
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from document_processor import load_documents, chunk_documents, DOCUMENTS_DIR # Import from your previous script

# Load environment variables (AWS credentials and region)
load_dotenv()

# --- Configuration ---
AWS_REGION = os.getenv("AWS_REGION")
# Choose an embedding model available in Bedrock for your region
# Common choices: "amazon.titan-embed-text-v1", "cohere.embed-english-v3"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1" 
FAISS_INDEX_PATH = "faiss_index" # Directory to save the FAISS index

def get_bedrock_embeddings_client():
    """Initializes and returns a BedrockEmbeddings client."""
    # Check if AWS region is available, fallback if not (as observed previously)
    if AWS_REGION:
        print(f"Initializing BedrockEmbeddings client for region: {AWS_REGION} using model: {EMBEDDING_MODEL_ID}")
        session = boto3.Session(
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        client = session.client(service_name="bedrock-runtime") # Use bedrock-runtime for invoking models

    else:
        print(f"AWS_REGION not set in .env. Initializing BedrockEmbeddings client with default region using model: {EMBEDDING_MODEL_ID}")
        # If AWS_REGION is None, boto3 will attempt to use default configured region or assume us-east-1
        # Still pass credentials explicitly for clarity
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        client = session.client(service_name="bedrock-runtime") # Use bedrock-runtime for invoking models


    # This is the LangChain wrapper for Bedrock embeddings
    embeddings_model = BedrockEmbeddings(
        client=client,
        model_id=EMBEDDING_MODEL_ID
    )
    return embeddings_model

def create_and_save_vector_store(chunks: list[Document], embeddings_client, index_path: str):
    """
    Creates a FAISS vector store from document chunks and saves it locally.
    """
    if not chunks:
        print("No chunks provided to create vector store. Exiting.")
        return None

    print(f"Creating FAISS vector store with {len(chunks)} chunks...")
    try:
        # Using from_documents will call the embedding model for each chunk
        vector_store = FAISS.from_documents(chunks, embeddings_client)
        print(f"FAISS vector store created successfully.")
        
        # Save the index to a local directory
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        vector_store.save_local(index_path)
        print(f"FAISS index saved to '{index_path}'.")
        return vector_store
    except Exception as e:
        print(f"Error creating or saving FAISS vector store: {e}")
        print("Please ensure your Bedrock embedding model is enabled/accessible in your region.")
        return None

if __name__ == "__main__":
    # 1. Load and chunk documents (re-using logic from document_processor.py)
    print("Starting document loading and chunking...")
    raw_documents = load_documents(DOCUMENTS_DIR)
    
    if not raw_documents:
        print("No documents found. Please ensure your 'data' directory contains .txt files.")
        exit() # Stop execution if no documents

    # Using the same chunking parameters as in step 5, adjust if needed
    processed_chunks = chunk_documents(raw_documents, chunk_size=700, chunk_overlap=150) 
    print(f"Finished chunking. Total chunks: {len(processed_chunks)}")

    # 2. Initialize Bedrock Embeddings Client
    bedrock_embeddings = get_bedrock_embeddings_client()
    if bedrock_embeddings is None:
        print("Failed to initialize Bedrock Embeddings client. Cannot proceed.")
        exit()

    # 3. Create and Save FAISS Vector Store
    vector_db = create_and_save_vector_store(processed_chunks, bedrock_embeddings, FAISS_INDEX_PATH)

    if vector_db:
        print("\nVector store successfully created and saved.")
        # Optional: Test loading the index to confirm it can be retrieved
        print(f"Attempting to load index from '{FAISS_INDEX_PATH}' for verification...")
        try:
            loaded_vector_db = FAISS.load_local(FAISS_INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True) # allow_dangerous_deserialization is needed for newer FAISS versions
            print("FAISS index loaded successfully from disk.")
            # You can perform a test search here if you want:
            # query = "What was the total revenue in Q3 2024 for TechGrowth?"
            # docs = loaded_vector_db.similarity_search(query, k=2)
            # print(f"\nTest search for '{query[:50]}...':")
            # for doc in docs:
            #    print(f"- Source: {doc.metadata.get('source')}, Content: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error loading FAISS index from disk: {e}")