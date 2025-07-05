# RAG-POC (Retrieval-Augmented Generation Proof of Concept)

This repository contains a production-ready implementation for Retrieval-Augmented Generation (RAG) with PDF document processing.

## Features

- Extract text from PDF documents using PyPDF2
- Process documents in parallel for improved performance
- Create semantic chunks for Q&A and summary purposes
- Generate vector embeddings using sentence-transformers
- Store embeddings in ChromaDB for efficient vector search
- Save processed chunks as JSON files for downstream applications

## Project Structure

- `extract_pdf_text.py`: PDF text extraction with parallel processing support
- `chunker.py`: Text chunking with semantic sentence-based chunking for Q&A and word-based chunking for summaries
- `embedder.py`: Vector embedding generation using sentence-transformers and ChromaDB integration
- `main.py`: Main script to process PDFs, generate chunks, and create embeddings

## Usage

1. Place PDF files in the `pdfs` directory
2. Run the main script:

```bash
# Standard usage (with embeddings - recommended for production)
python main.py --save

# Process PDFs in parallel with embeddings
python main.py --parallel --save

# Skip embeddings (not recommended for production use)
python main.py --save --skip_embed

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
pip install -r requirements.txt
```

## Embedding Model

The system uses the `all-MiniLM-L6-v2` model from sentence-transformers, which provides:
- 384-dimensional embeddings
- Good balance between quality and performance
- Multilingual support

## Vector Database

ChromaDB is used as the vector database for storing and retrieving embeddings. The embeddings are stored in:
- Collection "qna_chunks" for question-answering chunks
- Collection "summary_chunks" for summary chunks

## Chunk Sizes

- Q&A chunks: 300 words with 1 sentence overlap
- Summary chunks: 1000 words with no overlap

## Performance

The parallel processing option provides approximately 2x speedup compared to sequential processing.

## Compatibility

This implementation is compatible with:
- Python 3.9+
- NumPy 2.0+
- ChromaDB 1.0+
- Latest sentence-transformers 