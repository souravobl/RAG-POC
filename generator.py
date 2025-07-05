# generator.py

import os
import time
from llama_cpp import Llama
from retriever import retrieve_chunks

# === Config ===
MODEL_PATH = "llm_model/mistral-7b-instruct-v0.1.Q4_0.gguf"
MAX_TOKENS = 512  # Limit generation to avoid slow long responses

# Auto-select thread count based on CPU
import multiprocessing
CPU_THREADS = max(2, multiprocessing.cpu_count() - 1)

# Global LLM instance - load once, use many times
_LLM_INSTANCE = None

# === Load GGUF Model ===
def load_model():
    global _LLM_INSTANCE
    
    # Return existing instance if already loaded
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    print(f"🧠 Loading model from: {MODEL_PATH}")
    print(f"Using {CPU_THREADS} CPU threads")
    
    _LLM_INSTANCE = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=CPU_THREADS,
        n_batch=512,  # Increased batch size for faster processing
        use_mlock=True,  # lock model in RAM to avoid swap
        verbose=False
    )
    return _LLM_INSTANCE




# === Build Prompt ===
def build_prompt(query: str, chunks: list, task: str = "qa") -> str:
    # Limit the amount of context to avoid exceeding token limits
    # Estimate roughly 4 chars per token
    max_context_chars = 1200  # Roughly 300 tokens for context
    
    # Build context with length tracking
    context_parts = []
    total_chars = 0
    
    for chunk in chunks:
        chunk_text = f"[Source: {chunk['metadata']['pdf_name']} - Page {chunk['metadata']['page_number']}]\n{chunk['text']}"
        chunk_chars = len(chunk_text)
        
        # If adding this chunk would exceed our limit, stop
        if total_chars + chunk_chars > max_context_chars:
            if not context_parts:  # If we haven't added any chunks yet, add a truncated version
                truncated_text = chunk_text[:max_context_chars]
                context_parts.append(truncated_text)
            break
        
        # Otherwise add the chunk and update our counter
        context_parts.append(chunk_text)
        total_chars += chunk_chars
    
    context = "\n\n".join(context_parts)
    
    if task == "summary":
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful expert assistant. Your task is to create a comprehensive summary of the document based on the provided content.
Focus on the main topics, key points, and important information. Organize the summary in a clear, structured way.
<</SYS>>

Document Content:
{context}

Task: {query}
Summary: [/INST]"""
    else:  # Default to QA
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful expert assistant. Answer the question strictly based on the provided context.
<</SYS>>

Context:
{context}

Question: {query}
Answer: [/INST]"""
    
    return prompt


# === Main Answer Generator ===
def generate_answer(query: str, top_k: int = 5, filter_criteria=None, task: str = "qa", chunks_override=None):
    # Start timing the entire process
    total_start_time = time.time()
    
    # Step 1: Retrieve relevant chunks (unless overridden)
    if chunks_override is not None:
        print("🔍 Using provided chunks...")
        retrieved_chunks = chunks_override
        print(f"  - Using {len(retrieved_chunks)} provided chunks")
    else:
        print("🔍 Retrieving top relevant chunks...")
        retrieval_start = time.time()
        retrieved_chunks = retrieve_chunks(query, task="qna" if task == "qa" else "summary", top_k=top_k, filter_criteria=filter_criteria)
        retrieval_time = time.time() - retrieval_start
        print(f"  - Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f} seconds")

    # Step 2: Build the prompt
    print("📜 Constructing prompt...")
    prompt_start = time.time()
    prompt = build_prompt(query, retrieved_chunks, task=task)
    prompt_time = time.time() - prompt_start
    print(f"  - Prompt constructed in {prompt_time:.2f} seconds")

    # Step 3: Run inference
    print("🤖 Running inference using Mistral...")
    inference_start = time.time()
    
    # Get the model (will be loaded only once)
    llm = load_model()
    
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.2,
        stop=["</s>"]
    )
    inference_time = time.time() - inference_start
    print(f"  - Inference completed in {inference_time:.2f} seconds")

    answer = response["choices"][0]["text"].strip()
    
    # Report total time
    total_time = time.time() - total_start_time
    print(f"  - Total generation time: {total_time:.2f} seconds")

    return answer, retrieved_chunks
