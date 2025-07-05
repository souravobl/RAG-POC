import os
import json
import argparse
import time
from extract_pdf_text import extract_pdf_text, extract_pdf_text_batch
from chunker import chunk_text_dual
import pprint

def process_pdf(pdf_path, pdf_file):
    """Process a single PDF file."""
    start_time = time.time()
    print(f"\n🔍 Extracting text from: {pdf_file}")
    pdf_text = extract_pdf_text(pdf_path)
    extract_time = time.time() - start_time
    print(f"✅ Extracted {len(pdf_text)} pages in {extract_time:.2f} seconds.")

    print(f"🔧 Chunking {pdf_file} into Q&A and Summary...")
    chunk_start = time.time()
    chunks = chunk_text_dual(pdf_text, source_name=pdf_file)
    chunk_time = time.time() - chunk_start
    
    print(f"  - Q&A chunks: {len(chunks['qna'])} (Avg words: {sum(c.get('word_count', 0) for c in chunks['qna']) / max(1, len(chunks['qna'])):.1f})")
    print(f"  - Summary chunks: {len(chunks['summary'])} (Avg words: {sum(c.get('word_count', 0) for c in chunks['summary']) / max(1, len(chunks['summary'])):.1f})")
    print(f"  - Chunking completed in {chunk_time:.2f} seconds")
    
    return chunks

def save_chunks_to_file(chunks, output_dir, filename_prefix):
    """Save chunks to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save QnA chunks
    qna_path = os.path.join(output_dir, f"{filename_prefix}_qna_chunks.json")
    with open(qna_path, 'w') as f:
        json.dump(chunks['qna'], f, indent=2)
    
    # Save summary chunks
    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary_chunks.json")
    with open(summary_path, 'w') as f:
        json.dump(chunks['summary'], f, indent=2)
    
    return qna_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Process PDFs into chunks for RAG')
    parser.add_argument('--pdf_dir', default="pdfs", help='Directory containing PDF files')
    parser.add_argument('--output_dir', default="chunks", help='Directory to save chunks')
    parser.add_argument('--save', action='store_true', help='Save chunks to files')
    parser.add_argument('--parallel', action='store_true', help='Process PDFs in parallel')
    args = parser.parse_args()
    
    pdf_directory = args.pdf_dir
    output_directory = args.output_dir
    
    # Get list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    all_qna_chunks = []
    all_summary_chunks = []
    total_start_time = time.time()
    
    if args.parallel:
        # Process all PDFs in parallel
        pdf_paths = [os.path.join(pdf_directory, pdf_file) for pdf_file in pdf_files]
        print(f"🚀 Processing {len(pdf_files)} PDFs in parallel...")
        
        # Extract text from all PDFs in parallel
        start_time = time.time()
        pdf_texts = extract_pdf_text_batch(pdf_paths)
        extract_time = time.time() - start_time
        print(f"✅ Extracted text from {len(pdf_texts)} PDFs in {extract_time:.2f} seconds.")
        
        # Process each PDF's text
        for pdf_file, pdf_text in pdf_texts.items():
            if not pdf_text:  # Skip if extraction failed
                continue
                
            print(f"\n🔧 Chunking {pdf_file} into Q&A and Summary...")
            chunk_start = time.time()
            chunks = chunk_text_dual(pdf_text, source_name=pdf_file)
            chunk_time = time.time() - chunk_start
            
            all_qna_chunks.extend(chunks['qna'])
            all_summary_chunks.extend(chunks['summary'])
            
            print(f"  - Q&A chunks: {len(chunks['qna'])} (Avg words: {sum(c.get('word_count', 0) for c in chunks['qna']) / max(1, len(chunks['qna'])):.1f})")
            print(f"  - Summary chunks: {len(chunks['summary'])} (Avg words: {sum(c.get('word_count', 0) for c in chunks['summary']) / max(1, len(chunks['summary'])):.1f})")
            print(f"  - Chunking completed in {chunk_time:.2f} seconds")
            
            # Save individual PDF chunks if requested
            if args.save:
                filename_prefix = os.path.splitext(pdf_file)[0]
                qna_path, summary_path = save_chunks_to_file(
                    {'qna': chunks['qna'], 'summary': chunks['summary']},
                    output_directory,
                    filename_prefix
                )
                print(f"  - Saved chunks to {qna_path} and {summary_path}")
    else:
        # Process PDFs sequentially
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            chunks = process_pdf(pdf_path, pdf_file)
            
            all_qna_chunks.extend(chunks['qna'])
            all_summary_chunks.extend(chunks['summary'])
            
            # Save individual PDF chunks if requested
            if args.save:
                filename_prefix = os.path.splitext(pdf_file)[0]
                qna_path, summary_path = save_chunks_to_file(
                    {'qna': chunks['qna'], 'summary': chunks['summary']},
                    output_directory,
                    filename_prefix
                )
                print(f"  - Saved chunks to {qna_path} and {summary_path}")
    
    total_time = time.time() - total_start_time
    
    print(f"\n📊 Total results:")
    print(f"  - Total PDF files processed: {len(pdf_files)}")
    print(f"  - Total Q&A chunks: {len(all_qna_chunks)}")
    print(f"  - Total Summary chunks: {len(all_summary_chunks)}")
    print(f"  - Total processing time: {total_time:.2f} seconds")

    # Save combined chunks if requested
    if args.save:
        combined_chunks = {
            'qna': all_qna_chunks,
            'summary': all_summary_chunks
        }
        qna_path, summary_path = save_chunks_to_file(combined_chunks, output_directory, "combined")
        print(f"  - Saved combined chunks to {qna_path} and {summary_path}")

    print("\n--- Sample Q&A Chunk ---")
    if all_qna_chunks:
        pprint.pprint(all_qna_chunks[0])

    print("\n--- Sample Summary Chunk ---")
    if all_summary_chunks:
        pprint.pprint(all_summary_chunks[0])

if __name__ == "__main__":
    main()
