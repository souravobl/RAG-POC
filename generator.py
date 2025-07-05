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


# === Load GGUF Model ===
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    print(f"🧠 Loading model from: {MODEL_PATH}")
    print(CPU_THREADS)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=CPU_THREADS,
        n_batch=64,
        use_mlock=True,  # lock model in RAM to avoid swap
        verbose=False
    )
    return llm




# === Build Prompt ===
def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[Source: {c['metadata']['pdf_name']} - Page {c['metadata']['page_number']}]\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""<s>[INST] <<SYS>>
You are a helpful expert assistant. Answer the question strictly based on the provided context.
<</SYS>>

Context:
{context}

Question: {query}
Answer: [/INST]"""
    return prompt


# === Main Answer Generator ===
def generate_answer(query: str, top_k: int = 4, filter_criteria=None):
    # Start timing the entire process
    total_start_time = time.time()
    
    # Step 1: Retrieve relevant chunks
    print("🔍 Retrieving top relevant chunks...")
    retrieval_start = time.time()
    retrieved_chunks = retrieve_chunks(query, task="qna", top_k=top_k, filter_criteria=filter_criteria)
    retrieval_time = time.time() - retrieval_start
    print(f"  - Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f} seconds")

    # Step 2: Build the prompt
    print("📜 Constructing prompt...")
    prompt_start = time.time()
    prompt = build_prompt(query, retrieved_chunks)
    prompt_time = time.time() - prompt_start
    print(f"  - Prompt constructed in {prompt_time:.2f} seconds")

    # Step 3: Run inference
    print("🤖 Running inference using Mistral...")
    inference_start = time.time()
    llm = load_model()
    
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
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
