# retriever.py

import chromadb
import time

# Paths and collection names
CHROMA_DIR = "chroma_db"
QNA_COLLECTION_NAME = "qna_chunks"
SUMMARY_COLLECTION_NAME = "summary_chunks"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Cache for collections to avoid repeated lookups
_collection_cache = {}

def retrieve_chunks(query, task="qna", top_k=3, filter_criteria=None):
    """
    Retrieve chunks from ChromaDB based on a query.
    
    Args:
        query (str): The search query
        task (str): Either "qna" or "summary" to specify which collection to use
        top_k (int): Number of results to return
        filter_criteria (dict, optional): Filter criteria for metadata
        
    Returns:
        list: List of dictionaries with retrieved chunks and metadata
    """
    start_time = time.time()
    
    # Validate task parameter
    if task not in ["qna", "summary"]:
        raise ValueError("Task must be either 'qna' or 'summary'")
    
    # Get the appropriate collection (using cache if available)
    collection_name = f"{task}_chunks"
    
    if collection_name in _collection_cache:
        collection = _collection_cache[collection_name]
    else:
        collection = client.get_collection(collection_name)
        _collection_cache[collection_name] = collection
    
    # Prepare query parameters
    query_params = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    
    # Add filter if provided - ChromaDB requires a specific format with operators
    if filter_criteria:
        # ChromaDB requires compound filters to use $and operator
        if len(filter_criteria) > 1:
            # Create a list of conditions for the $and operator
            conditions = []
            for key, value in filter_criteria.items():
                conditions.append({key: {"$eq": value}})
            
            # Use the $and operator for multiple conditions
            query_params["where"] = {"$and": conditions}
        else:
            # For single condition, use simple format
            key, value = next(iter(filter_criteria.items()))
            query_params["where"] = {key: {"$eq": value}}
    
    # Execute query
    try:
        results = collection.query(**query_params)
    except Exception as e:
        print(f"Query error: {str(e)}")
        print(f"Query parameters: {query_params}")
        raise
    
    # Format results
    formatted_results = []
    
    # Handle case where no results are found
    if not results["documents"] or len(results["documents"][0]) == 0:
        query_time = time.time() - start_time
        print(f"  - No results found (search took {query_time:.4f} seconds)")
        return []
        
    for i in range(len(results["documents"][0])):
        # Convert distance to similarity score (1 - normalized_distance)
        # ChromaDB distances are already normalized between 0-1
        similarity_score = 1 - results["distances"][0][i]
        
        formatted_results.append({
            "text": results["documents"][0][i],
            "metadata": {
                "pdf_name": results["metadatas"][0][i]["source"],
                "page_number": results["metadatas"][0][i]["page_num"],
                "chunk_type": results["metadatas"][0][i]["chunk_type"]
            },
            "score": similarity_score
        })
    
    query_time = time.time() - start_time
    print(f"  - Retrieved {len(formatted_results)} chunks in {query_time:.4f} seconds")
    
    return formatted_results
