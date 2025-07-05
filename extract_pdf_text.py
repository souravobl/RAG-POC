from typing import Dict
import os
from PyPDF2 import PdfReader
from multiprocessing import Pool, cpu_count

def extract_text_from_single_pdf(pdf_path: str) -> Dict[int, str]:
    """
    Extracts text from each page of a single PDF.
    
    Returns:
        Dict[page_num, text]
    """
    try:
        reader = PdfReader(pdf_path)
        text_by_page = {}
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_by_page[i + 1] = text
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1} in {os.path.basename(pdf_path)}: {str(e)}")
        
        return text_by_page
    except Exception as e:
        print(f"Error processing PDF {os.path.basename(pdf_path)}: {str(e)}")
        return {}

def extract_pdf_text(pdf_path: str) -> Dict[int, str]:
    """
    Extracts text from each page of the PDF.
    
    Returns:
        Dict[page_num, text]
    """
    # For single file processing, just call the function directly
    return extract_text_from_single_pdf(pdf_path)

def extract_pdf_text_batch(pdf_paths: list) -> Dict[str, Dict[int, str]]:
    """
    Extracts text from multiple PDFs in parallel using multiprocessing.
    
    Args:
        pdf_paths: List of PDF file paths
        
    Returns:
        Dict[pdf_name, Dict[page_num, text]]
    """
    # Determine optimal number of processes (use at most cpu_count)
    num_processes = min(len(pdf_paths), cpu_count())
    
    # Use multiprocessing to extract text from PDFs in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(extract_text_from_single_pdf, pdf_paths)
    
    # Combine results into a dictionary with filenames as keys
    return {os.path.basename(path): result for path, result in zip(pdf_paths, results)}
