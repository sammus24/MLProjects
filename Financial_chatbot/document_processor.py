import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the directory where your documents are stored
DOCUMENTS_DIR = "data"

def load_documents(directory):
    """Loads all text documents from the specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "filename": filename,
                    "content": content
                })
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """Chunks the content of documents using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Order matters: try splitting by paragraphs first
    )
    
    all_chunks = []
    for doc in documents:
        print(f"--- Processing {doc['filename']} ---")
        chunks = text_splitter.create_documents([doc['content']])
        # Add metadata to each chunk for easier tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata = {"source": doc['filename'], "chunk_id": i}
        all_chunks.extend(chunks)
        print(f"Created {len(chunks)} chunks for {doc['filename']}")
    return all_chunks

if __name__ == "__main__":
    print(f"Loading documents from {DOCUMENTS_DIR}...")
    raw_documents = load_documents(DOCUMENTS_DIR)

    if not raw_documents:
        print(f"No documents found in '{DOCUMENTS_DIR}'. Please ensure your .txt files are there.")
    else:
        print(f"Found {len(raw_documents)} documents.")
        
        # Experiment with chunk_size and chunk_overlap here
        # For tabular data, you might want larger chunks or less overlap initially
        # to keep related rows together, then refine.
        processed_chunks = chunk_documents(raw_documents, chunk_size=700, chunk_overlap=150) # Increased chunk size for finance docs

        print(f"\nTotal chunks generated: {len(processed_chunks)}")
        print("\n--- Example Chunk (First 200 chars): ---")
        if processed_chunks:
            # Print a few example chunks to inspect their quality
            for i, chunk in enumerate(processed_chunks[:3]): # Print first 3 chunks
                print(f"\nChunk {i+1} (Source: {chunk.metadata.get('source')}):\n'{chunk.page_content[:200]}...'")
                print(f"Length: {len(chunk.page_content)} characters")
        else:
            print("No chunks were generated.")