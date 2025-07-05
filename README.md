# RAG-POC (Retrieval-Augmented Generation Proof of Concept)

This repository contains a proof of concept implementation for Retrieval-Augmented Generation (RAG) with PDF document processing.

## Features

- Extract text from PDF documents using PyPDF2
- Process documents in parallel for improved performance
- Create semantic chunks for Q&A and summary purposes
- Save processed chunks as JSON files for downstream RAG applications

## Project Structure

- `extract_pdf_text.py`: PDF text extraction with parallel processing support
- `chunker.py`: Text chunking with semantic sentence-based chunking for Q&A and word-based chunking for summaries
- `main.py`: Main script to process PDFs and generate chunks

## Usage

1. Place PDF files in the `pdfs` directory
2. Run the main script:

```bash
# Basic usage
python main.py

# Process PDFs in parallel and save chunks
python main.py --parallel --save

# Specify custom directories
python main.py --pdf_dir custom_pdfs --output_dir custom_output
```

## Installation

```bash
# Create a virtual environment
python -m venv rag-env

# Activate the virtual environment
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate

# Install dependencies
pip install PyPDF2 nltk
```

## Chunk Sizes

- Q&A chunks: 300 words with 1 sentence overlap
- Summary chunks: 1000 words with no overlap

## Performance

The parallel processing option provides approximately 2x speedup compared to sequential processing. 