import os
import boto3
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock # Or from langchain_aws.chat_models import ChatBedrock for chat models

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough # For potentially chaining later if needed


# Load environment variables
load_dotenv()

# --- Configuration ---
AWS_REGION = os.getenv("AWS_REGION")
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_ID = "amazon.titan-text-express-v1"

# --- Functions for building the RAG chain ---

def get_bedrock_llm_client():
    """Initializes and returns a Bedrock LLM client."""
    # Use Bedrock LLM or ChatBedrock based on your needs.
    # ChatBedrock is recommended for newer models and chat-like interactions.
    # If using ChatBedrock, import from langchain_aws.chat_models
    
    # Ensure the client session uses explicit credentials
    session = boto3.Session(
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    client = session.client(service_name="bedrock-runtime")

    print(f"Initializing Bedrock LLM client for region: {AWS_REGION} using model: {LLM_MODEL_ID}")
    
    llm = Bedrock( # For chat models: llm = ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={
            "temperature": 0.1,
            "maxTokenCount": 500
        }
    )
    return llm

def get_bedrock_embeddings_client():
    """Initializes and returns a BedrockEmbeddings client."""
    # Ensure the client session uses explicit credentials
    session = boto3.Session(
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    client = session.client(service_name="bedrock-runtime") # Use bedrock-runtime for invoking models

    embeddings_model = BedrockEmbeddings(
        client=client,
        model_id=EMBEDDING_MODEL_ID
    )
    return embeddings_model

def load_vector_store(index_path: str, embeddings_client):
    """Loads a FAISS vector store from a local directory."""
    if not os.path.exists(index_path):
        print(f"Error: FAISS index directory '{index_path}' not found. Please run vector_store_manager.py first.")
        return None
    
    print(f"Loading FAISS index from '{index_path}'...")
    try:
        vector_store = FAISS.load_local(index_path, embeddings_client, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def create_rag_chain():
    """
    Creates and returns the full RAG retrieval chain.
    """
    print("Creating RAG chain...")
    
    # Get embeddings client for vector store loading
    bedrock_embeddings = get_bedrock_embeddings_client()
    if bedrock_embeddings is None:
        return None

    # Load FAISS Vector Store
    vector_db = load_vector_store(FAISS_INDEX_PATH, bedrock_embeddings)
    if vector_db is None:
        return None

    # Initialize Bedrock LLM Client
    llm = get_bedrock_llm_client()
    if llm is None:
        return None

    # Define the Prompt Template
    # This prompt is crucial for instructing the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI assistant specialized in providing accurate information from financial documents. "
         "Answer the user's question ONLY based on the provided context. "
         "If the answer cannot be found in the context, clearly state that you don't have enough information. "
         "Do not make up information. Prioritize factual accuracy. \n\n"
         "Context: {context}"),
        ("human", "{input}")
    ])

    # Create the document combining chain (stuffing documents into the prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the RAG retrieval chain
    retriever = vector_db.as_retriever()
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain created successfully.")
    return retrieval_chain

# The __main__ block will be removed or commented out for Streamlit
# if __name__ == "__main__":
#     # Your interactive terminal code from before
#     rag_chain = create_rag_chain()
#     if rag_chain:
#         print("\nAI Financial Assistant Ready! Type your queries below. Type 'exit' to quit.")
#         while True:
#             user_query = input("\nYour Query: ")
#             if user_query.lower() == 'exit':
#                 print("Exiting assistant. Goodbye!")
#                 break
#             try:
#                 response = rag_chain.invoke({"input": user_query})
#                 print("\nAssistant:")
#                 print(response["answer"])
#                 print("\n--- Sources Used ---")
#                 if response["context"]:
#                     for i, doc in enumerate(response["context"]):
#                         print(f"Chunk {i+1} from {doc.metadata.get('source')}:")
#                         print(f"Content: {doc.page_content[:150]}...")
#                 else:
#                     print("No specific source documents were retrieved for this query.")
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 print("Please ensure your Bedrock LLM model is enabled/accessible and your internet connection is stable.")