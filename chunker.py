import re
from typing import Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

def clean_text(text: str) -> str:
    """
    Clean and normalize text for better chunking.
    """
    # Replace multiple newlines, tabs and spaces with single space
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    # Additional cleaning
    text = text.replace('…', '...')  # Normalize ellipsis
    
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex for simplicity.
    """
    # Simple sentence splitting using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def _chunk_page_by_words(
    words: List[str],
    page_num: int,
    source_name: str,
    chunk_size: int,
    overlap: int,
    chunk_type: str
) -> List[Dict]:
    """
    Chunk by word count (original method).
    """
    chunks = []
    i = 0
    chunk_index = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "source": source_name,
            "page_num": page_num,
            "chunk_index": chunk_index,
            "type": chunk_type,
            "text": chunk_text,
            "word_count": len(chunk_words)
        })

        i += chunk_size - overlap if chunk_size > overlap else chunk_size
        chunk_index += 1

    return chunks

def _chunk_page_by_sentences(
    sentences: List[str],
    page_num: int,
    source_name: str,
    target_size: int,
    overlap_sentences: int,
    chunk_type: str
) -> List[Dict]:
    """
    Chunk by sentences for more semantic coherence.
    """
    chunks = []
    i = 0
    chunk_index = 0
    
    while i < len(sentences):
        # Start with one sentence
        current_chunk = [sentences[i]]
        current_size = len(current_chunk[0].split())
        j = i + 1
        
        # Add sentences until we reach target size
        while j < len(sentences) and current_size < target_size:
            current_chunk.append(sentences[j])
            current_size += len(sentences[j].split())
            j += 1
            
        chunk_text = " ".join(current_chunk)
        
        chunks.append({
            "source": source_name,
            "page_num": page_num,
            "chunk_index": chunk_index,
            "type": chunk_type,
            "text": chunk_text,
            "word_count": current_size,
            "sentence_count": len(current_chunk)
        })
        
        # Move forward with overlap
        i += max(1, len(current_chunk) - overlap_sentences)
        chunk_index += 1
        
    return chunks

def process_page(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single page for threading.
    """
    page_num, text, source_name, qna_size, qna_overlap, summary_size, summary_overlap = args
    
    clean = clean_text(text)
    
    # For Q&A, use sentence-based chunking for better semantic units
    sentences = split_into_sentences(clean)
    qna_chunks = _chunk_page_by_sentences(
        sentences, page_num, source_name, qna_size, 1, "qna"
    )
    
    # For summaries, use word-based chunking to get larger, fixed-size chunks
    words = clean.split()
    summary_chunks = _chunk_page_by_words(
        words, page_num, source_name, summary_size, summary_overlap, "summary"
    )
    
    return qna_chunks, summary_chunks

def chunk_text_dual(
    pdf_text: Dict[int, str],
    source_name: str,
    qna_size: int = 300,
    qna_overlap: int = 1,  # Number of sentences to overlap
    summary_size: int = 1000,
    summary_overlap: int = 0
) -> Dict[str, List[Dict]]:
    """
    Returns two separate chunk lists: one for Q&A and one for Summary.
    Uses multithreading for faster processing.
    """
    # Prepare arguments for each page
    args = [
        (page_num, text, source_name, qna_size, qna_overlap, summary_size, summary_overlap)
        for page_num, text in pdf_text.items()
    ]
    
    qna_chunks = []
    summary_chunks = []
    
    # Use ThreadPoolExecutor for parallel processing of pages
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, len(pdf_text))) as executor:
        results = list(executor.map(process_page, args))
        
    # Combine results
    for qna, summary in results:
        qna_chunks.extend(qna)
        summary_chunks.extend(summary)

    return {
        "qna": qna_chunks,
        "summary": summary_chunks
    }
