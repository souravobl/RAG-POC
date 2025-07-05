# embedder_v2.py - Compatible with latest ChromaDB and NumPy 2.0+

import os
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

def embed_chunks_to_chroma(model, collection_name, chunks):
    client = get_chroma_client()
    
    # Delete collection if it exists to avoid duplicates
    try:
        client.delete_collection(name=collection_name)
        print(f"[!] Deleted existing collection: {collection_name}")
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)

    texts, metadatas, ids = [], [], []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        metadata = {
            "source": chunk["source"],
            "page_num": chunk["page_num"],
            "chunk_type": chunk["type"]
        }
        texts.append(text)
        metadatas.append(metadata)
        ids.append(f"{collection_name}_{i}")

    print(f"[+] Embedding {len(texts)} chunks into collection: {collection_name}")
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

def embed_chunks(qna_chunks, summary_chunks):
    model = load_embedding_model()
    qna_count = embed_chunks_to_chroma(model, "qna_chunks", qna_chunks)
    summary_count = embed_chunks_to_chroma(model, "summary_chunks", summary_chunks)
    
    total_count = qna_count + summary_count
    print(f"[✓] Total embeddings created: {total_count}")
    
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
        "total_count": total_count
    } 