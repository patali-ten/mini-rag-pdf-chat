# PDF Question Answering RAG

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a PDF document using semantic search and Google's Gemini models.

## Features

* Extract text from PDF documents
* Split documents into manageable chunks
* Generate embeddings using Gemini Embedding Model
* Store embeddings in a FAISS vector index
* Retrieve the most relevant chunks using semantic similarity
* Generate grounded answers with Gemini 2.5 Flash
* Reduce hallucinations by restricting answers to retrieved context
* Secure API key management using environment variables

## Tech Stack

* Python
* Gemini 2.5 Flash
* Gemini Embedding Model
* FAISS
* PyMuPDF
* NumPy
* python-dotenv

## Architecture

PDF Document
    ↓
Text Extraction
    ↓
Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Search
    ↓
Relevant Context Retrieval
    ↓
Gemini 2.5 Flash
    ↓
Answer Generation

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create a .env file

```env
GEMINI_API_KEY=your_api_key_here
```

4. Place your PDF in the project directory

```text
sample.pdf
```

5. Run the application

```bash
python rag.py
```

## Example Workflow

1. Load a PDF document
2. Convert the document into chunks
3. Generate embeddings for each chunk
4. Build a FAISS vector index
5. Ask a question
6. Retrieve the most relevant chunks
7. Generate an answer using Gemini

## Future Improvements

* Persistent FAISS index storage
* Multi-document support
* Hybrid search (keyword + vector search)
* Web interface with Streamlit
* Conversation memory
* Advanced reranking models

## Learning Objectives

This project demonstrates the core concepts behind modern Retrieval-Augmented Generation (RAG) systems:

* Embeddings
* Vector Databases
* Semantic Search
* Retrieval Pipelines
* Grounded LLM Responses
