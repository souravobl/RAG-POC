# main.py

import os
import json
import argparse
import time
import pprint
from extract_pdf_text import extract_pdf_text, extract_pdf_text_batch
from chunker import chunk_text_dual
from embedder import embed_chunks

# Import the generator for interactive Q&A
try:
    from generator import generate_answer
    GENERATOR_AVAILABLE = True
except ImportError:
    print("Note: LLM generator module not available. Interactive Q&A will be disabled.")
    GENERATOR_AVAILABLE = False

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
    
    qna_path = os.path.join(output_dir, f"{filename_prefix}_qna_chunks.json")
    with open(qna_path, 'w') as f:
        json.dump(chunks['qna'], f, indent=2)
    
    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary_chunks.json")
    with open(summary_path, 'w') as f:
        json.dump(chunks['summary'], f, indent=2)
    
    return qna_path, summary_path

def get_available_pdfs():
    """Get list of available PDFs in the pdfs directory."""
    pdf_dir = "pdfs"
    if not os.path.exists(pdf_dir):
        return []
    return [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

def display_menu(options, title="Menu"):
    """Display a menu and get user selection."""
    print(f"\n{title}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options):
                return choice
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")

def interactive_qa_mode(pdf_filter=None):
    """
    Run interactive Q&A mode with the user.
    
    Args:
        pdf_filter: If provided, only search in this specific PDF
    """
    if not GENERATOR_AVAILABLE:
        print("\n❌ Interactive Q&A requires the generator module.")
        print("Please ensure you have the necessary LLM model installed.")
        return

    pdf_name = pdf_filter if pdf_filter else "All Knowledge Base"
    
    print("\n" + "=" * 50)
    print(f"🤖 Interactive Q&A Mode - {pdf_name}")
    print("=" * 50)
    print("Type your questions and get answers from the processed documents.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'menu' to return to the main menu.")
    
    while True:
        print("\n" + "-" * 50)
        query = input("🔍 Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q', '']:
            print("Ending Q&A session. Goodbye!")
            break
            
        if query.lower() == 'menu':
            return
            
        try:
            # Start timing the entire process
            total_start_time = time.time()
            
            print("\nGenerating answer...")
            
            # Add filter criteria if a specific PDF was selected
            filter_criteria = {"source": pdf_filter} if pdf_filter else None
            answer, sources = generate_answer(query, filter_criteria=filter_criteria)
            
            # Calculate total time taken
            total_time = time.time() - total_start_time
            
            print("\n" + "=" * 50)
            print("📝 Answer:")
            print(answer)
            
            print("\n📚 Retrieved Chunks:")
            for i, source in enumerate(sources, 1):
                print(f"\n  {i}. Source: {source['metadata']['pdf_name']} (Page {source['metadata']['page_number']})")
                print(f"     Similarity Score: {source['score']:.4f}")
                # Print a preview of the chunk text (first 100 characters)
                preview = source['text'][:100] + "..." if len(source['text']) > 100 else source['text']
                print(f"     Preview: {preview}")
            
            print("\n⏱️ Performance:")
            print(f"  - Total response time: {total_time:.2f} seconds")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")

def get_pdf_summary(pdf_name):
    """
    Retrieve and display a summary for a specific PDF.
    
    Args:
        pdf_name: The name of the PDF to summarize
    """
    from retriever import retrieve_chunks
    
    try:
        print(f"\n🔍 Retrieving summary for {pdf_name}...")
        
        # Get summary chunks for this PDF
        filter_criteria = {"source": pdf_name, "chunk_type": "summary"}
        summary_chunks = retrieve_chunks(
            query="summary of document",  # Generic query to find summary chunks
            task="summary",
            top_k=50,  # Get more chunks to ensure we have all pages
            filter_criteria=filter_criteria
        )
        
        if not summary_chunks:
            print(f"\n❌ No summary chunks found for {pdf_name}")
            return
            
        print("\n" + "=" * 50)
        print(f"📚 Summary for {pdf_name}")
        print("=" * 50)
        
        # Sort chunks by page number
        summary_chunks.sort(key=lambda x: x['metadata']['page_number'])
        
        # First display the page-wise content
        print("\n📄 Page-by-page content:")
        for i, chunk in enumerate(summary_chunks, 1):
            page = chunk['metadata']['page_number']
            print(f"\n📄 Page {page}:")
            print("-" * 40)
            print(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
            print("-" * 40)
        
        # Now generate an actual summary using the LLM if available
        if GENERATOR_AVAILABLE:
            print("\n🤖 Generating an overall summary of the document...")
            try:
                # Create a query that asks for a summary
                query = f"Please provide a comprehensive summary of this document about {pdf_name}"
                
                # Limit the number of chunks to avoid context window issues
                # For longer documents, we'll summarize the first few pages
                max_chunks = 3  # Start with a small number of chunks
                
                if len(summary_chunks) > max_chunks:
                    print(f"\nℹ️ Document is long ({len(summary_chunks)} pages). Summarizing first {max_chunks} pages...")
                    limited_chunks = summary_chunks[:max_chunks]
                else:
                    limited_chunks = summary_chunks
                
                # Use the generator to create a summary based on the chunks
                filter_criteria = {"source": pdf_name, "chunk_type": "summary"}
                summary, _ = generate_answer(
                    query, 
                    filter_criteria=filter_criteria, 
                    top_k=max_chunks, 
                    task="summary",
                    chunks_override=limited_chunks  # Pass the chunks directly
                )
                
                print("\n" + "=" * 50)
                print("📝 Document Summary:")
                print("=" * 50)
                print(summary)
                print("=" * 50)
            except Exception as e:
                print(f"\n❌ Error generating summary: {str(e)}")
                print("Displaying raw chunks only.")
            
    except Exception as e:
        print(f"\n❌ Error retrieving summary: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed error for debugging

def interactive_menu():
    """Display the main interactive menu and handle user choices."""
    if not GENERATOR_AVAILABLE:
        print("\n❌ Interactive menu requires the generator module.")
        print("Please ensure you have the necessary LLM model installed.")
        return
        
    # Get available PDFs
    available_pdfs = get_available_pdfs()
    
    while True:
        print("\n" + "=" * 50)
        print("🤖 RAG System Interactive Menu")
        print("=" * 50)
        
        # Main menu options
        main_options = ["Q&A Mode", "Summary Mode", "Exit"]
        main_choice = display_menu(main_options, "Main Menu")
        
        if main_choice == 1:  # Q&A Mode
            # PDF selection for Q&A
            qa_options = ["All Knowledge Base"] + available_pdfs + ["Back to Main Menu"]
            qa_choice = display_menu(qa_options, "Select Knowledge Base for Q&A")
            
            if qa_choice == 1:  # All Knowledge Base
                interactive_qa_mode()
            elif qa_choice == len(qa_options):  # Back to Main Menu
                continue
            else:  # Specific PDF
                selected_pdf = available_pdfs[qa_choice - 2]
                interactive_qa_mode(pdf_filter=selected_pdf)
                
        elif main_choice == 2:  # Summary Mode
            # PDF selection for Summary
            summary_options = available_pdfs + ["Back to Main Menu"]
            summary_choice = display_menu(summary_options, "Select PDF for Summary")
            
            if summary_choice == len(summary_options):  # Back to Main Menu
                continue
            else:  # Specific PDF
                selected_pdf = available_pdfs[summary_choice - 1]
                get_pdf_summary(selected_pdf)
                input("\nPress Enter to continue...")
                
        elif main_choice == 3:  # Exit
            print("Exiting RAG System. Goodbye!")
            break

def main():
    # Define parser for backward compatibility, but use default settings
    parser = argparse.ArgumentParser(description='Process PDFs into chunks for RAG')
    parser.add_argument('--pdf_dir', default="pdfs", help='Directory containing PDF files')
    parser.add_argument('--output_dir', default="chunks", help='Directory to save chunks')
    parser.add_argument('--save', action='store_true', help='Save chunks to files')
    parser.add_argument('--parallel', action='store_true', help='Process PDFs in parallel')
    parser.add_argument('--skip_embed', action='store_true', help='Skip embedding step (not recommended)')
    parser.add_argument('--reset_embed', action='store_true', help='Reset embeddings (delete and recreate collections)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive Q&A mode after processing')
    parser.add_argument('--qa_only', action='store_true', help='Skip processing and go directly to Q&A mode')
    parser.add_argument('--menu', action='store_true', help='Start interactive menu mode')
    args = parser.parse_args()
    
    # Set default behavior: process PDFs in parallel, save chunks, and run interactive mode
    use_parallel = True  # Always use parallel processing for better performance
    save_chunks = True   # Always save chunks
    
    # If menu-only mode is selected, skip to interactive menu
    if args.menu:
        if GENERATOR_AVAILABLE:
            interactive_menu()
        else:
            print("❌ Interactive menu requires the generator module.")
        return
        
    # If QA-only mode is selected, skip to interactive menu
    if args.qa_only:
        if GENERATOR_AVAILABLE:
            interactive_menu()
        else:
            print("❌ Interactive menu requires the generator module.")
        return
    
    pdf_directory = args.pdf_dir
    output_directory = args.output_dir
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    all_qna_chunks = []
    all_summary_chunks = []
    total_start_time = time.time()
    
    print("\n" + "=" * 50)
    print("📚 STEP 1: Processing PDF Documents")
    print("=" * 50)
    
    if use_parallel:
        pdf_paths = [os.path.join(pdf_directory, pdf_file) for pdf_file in pdf_files]
        print(f"🚀 Processing {len(pdf_files)} PDFs in parallel...")
        
        start_time = time.time()
        pdf_texts = extract_pdf_text_batch(pdf_paths)
        extract_time = time.time() - start_time
        print(f"✅ Extracted text from {len(pdf_texts)} PDFs in {extract_time:.2f} seconds.")
        
        for pdf_file, pdf_text in pdf_texts.items():
            if not pdf_text:
                continue
                
            print(f"\n🔧 Chunking {pdf_file} into Q&A and Summary...")
            chunk_start = time.time()
            chunks = chunk_text_dual(pdf_text, source_name=pdf_file)
            chunk_time = time.time() - chunk_start
            
            all_qna_chunks.extend(chunks['qna'])
            all_summary_chunks.extend(chunks['summary'])
            
            print(f"  - Q&A chunks: {len(chunks['qna'])}")
            print(f"  - Summary chunks: {len(chunks['summary'])}")
            print(f"  - Chunking completed in {chunk_time:.2f} seconds")
            
            if save_chunks:
                filename_prefix = os.path.splitext(pdf_file)[0]
                qna_path, summary_path = save_chunks_to_file(
                    {'qna': chunks['qna'], 'summary': chunks['summary']},
                    output_directory,
                    filename_prefix
                )
                print(f"  - Saved chunks to {qna_path} and {summary_path}")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            chunks = process_pdf(pdf_path, pdf_file)
            
            all_qna_chunks.extend(chunks['qna'])
            all_summary_chunks.extend(chunks['summary'])
            
            if save_chunks:
                filename_prefix = os.path.splitext(pdf_file)[0]
                qna_path, summary_path = save_chunks_to_file(
                    {'qna': chunks['qna'], 'summary': chunks['summary']},
                    output_directory,
                    filename_prefix
                )
                print(f"  - Saved chunks to {qna_path} and {summary_path}")

    total_time = time.time() - total_start_time
    print(f"\n📊 Total PDFs processed: {len(pdf_files)}")
    print(f"  - Q&A chunks: {len(all_qna_chunks)}")
    print(f"  - Summary chunks: {len(all_summary_chunks)}")
    print(f"  - Total time: {total_time:.2f} seconds")

    if save_chunks:
        combined_chunks = {
            'qna': all_qna_chunks,
            'summary': all_summary_chunks
        }
        save_chunks_to_file(combined_chunks, output_directory, "combined")

    # Embedding step - always run by default unless skipped
    if not args.skip_embed:
        print("\n" + "=" * 50)
        print("🧠 STEP 2: Creating Vector Embeddings")
        print("=" * 50)
        
        try:
            print("🚀 Embedding chunks to ChromaDB...")
            embed_start_time = time.time()
            
            # Use reset flag if specified
            reset_embeddings = args.reset_embed
            if reset_embeddings:
                print("⚠️ Reset flag detected - collections will be deleted and recreated")
                
            embed_results = embed_chunks(all_qna_chunks, all_summary_chunks, reset=reset_embeddings)
            embed_time = time.time() - embed_start_time
            print(f"✅ Embedding complete in {embed_time:.2f} seconds.")
            print(f"  - Total embeddings: {embed_results['total_count']}")
        except Exception as e:
            print(f"\n❌ Critical Error: Embedding failed: {str(e)}")
            print("The RAG system cannot function properly without embeddings.")
            print("Please fix the embedding issues before proceeding.")
            return
    else:
        print("\n⚠️ WARNING: Embedding step skipped. Using existing embeddings.")

    # Always start interactive menu if generator is available
    if GENERATOR_AVAILABLE:
        print("\n" + "=" * 50)
        print("🤖 STEP 3: Interactive Mode")
        print("=" * 50)
        interactive_menu()
    else:
        print("\n❌ Interactive mode requires the generator module.")
        print("Please ensure you have the necessary LLM model installed.")

if __name__ == "__main__":
    main()
