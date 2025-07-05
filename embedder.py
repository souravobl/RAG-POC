# embedder.py - Compatible with latest ChromaDB and NumPy 2.0+

import os
import time
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Constants
MODEL_DIR = "model_cache"
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "chroma_db"

def load_embedding_model():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        print("[+] Downloading model...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(model_path)
    else:
        print("[+] Loading model from cache...")
        model = SentenceTransformer(model_path)
    return model

def get_chroma_client():
    # Use PersistentClient for the latest ChromaDB version
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def generate_chunk_id(chunk):
    """Generate a stable ID for a chunk based on its content and metadata."""
    content = f"{chunk['source']}_{chunk['page_num']}_{chunk['text'][:100]}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def embed_chunks_to_chroma(model, collection_name, chunks, reset=False):
    client = get_chroma_client()
    
    # Check if collection exists
    collection_exists = False
    try:
        existing_collections = client.list_collections()
        collection_exists = any(col.name == collection_name for col in existing_collections)
    except Exception as e:
        print(f"[!] Error checking collections: {str(e)}")
    
    # Handle collection creation/retrieval
    if reset and collection_exists:
        # Delete and recreate if reset is requested
        client.delete_collection(name=collection_name)
        print(f"[!] Deleted existing collection: {collection_name}")
        collection = client.create_collection(name=collection_name)
        print(f"[+] Created new collection: {collection_name}")
        existing_ids = []
    elif collection_exists:
        # Use existing collection
        collection = client.get_collection(name=collection_name)
        print(f"[+] Using existing collection: {collection_name}")
        
        # Get existing IDs to avoid duplicates
        try:
            # This is a simple approach - in a production system you might want
            # to page through results for very large collections
            existing_data = collection.get(include=['documents', 'metadatas', 'embeddings'])
            existing_ids = existing_data['ids']
            print(f"[+] Found {len(existing_ids)} existing documents")
        except Exception as e:
            print(f"[!] Error retrieving existing documents: {str(e)}")
            existing_ids = []
    else:
        # Create new collection
        collection = client.create_collection(name=collection_name)
        print(f"[+] Created new collection: {collection_name}")
        existing_ids = []

    # Process chunks and filter out any that might already exist
    texts, metadatas, ids = [], [], []
    new_chunks_count = 0

    for chunk in chunks:
        # Generate a stable ID for this chunk
        chunk_id = generate_chunk_id(chunk)
        
        # Skip if this chunk is already in the collection
        if chunk_id in existing_ids:
            continue
            
        new_chunks_count += 1
        text = chunk["text"]
        metadata = {
            "source": chunk["source"],
            "page_num": chunk["page_num"],
            "chunk_type": chunk["type"]
        }
        texts.append(text)
        metadatas.append(metadata)
        ids.append(chunk_id)

    # If no new chunks to add, we're done
    if new_chunks_count == 0:
        print(f"[✓] No new chunks to add to {collection_name}")
        count = collection.count()
        print(f"  - Collection contains {count} documents")
        return count
        
    print(f"[+] Embedding {len(texts)} new chunks into collection: {collection_name}")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert embeddings to list of lists for ChromaDB
    embeddings_list = embeddings.tolist()

    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            embeddings=embeddings_list[i:end_idx],
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"  - Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    print(f"[✓] Saved to ChromaDB: {collection_name}")
    
    # Verify the collection
    count = collection.count()
    print(f"  - Collection contains {count} documents")
    
    return count

def embed_chunks(qna_chunks, summary_chunks, reset=False):
    """
    Embed chunks into ChromaDB collections.
    
    Args:
        qna_chunks: List of chunks for Q&A
        summary_chunks: List of chunks for summaries
        reset: If True, delete existing collections and start fresh
    """
    start_time = time.time()
    model = load_embedding_model()
    
    qna_count = embed_chunks_to_chroma(model, "qna_chunks", qna_chunks, reset=reset)
    summary_count = embed_chunks_to_chroma(model, "summary_chunks", summary_chunks, reset=reset)
    
    total_count = qna_count + summary_count
    embed_time = time.time() - start_time
    print(f"[✓] Total embeddings: {total_count} (completed in {embed_time:.2f} seconds)")
    
    # Test a simple query to verify everything works
    try:
        client = get_chroma_client()
        collection = client.get_collection("qna_chunks")
        results = collection.query(
            query_texts=["What is electricity?"],
            n_results=1
        )
        print(f"[✓] Query test successful: found {len(results['documents'])} results")
    except Exception as e:
        print(f"[!] Query test failed: {str(e)}")
        
    return {
        "qna_count": qna_count,
        "summary_count": summary_count,
        "total_count": total_count,
        "time_taken": embed_time
    } 