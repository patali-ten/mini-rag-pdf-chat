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

```text
📄 PDF Document ➔ 🔍 Text Extraction ➔ ✂️ Chunking ➔ 🧬 Gemini Embeddings ➔ 🗄️ FAISS Index
                                                                              │
📝 Answer Generation 💾 💡◀─── ⚡ Gemini 2.5 Flash ◀─── 📥 Context Retrieval ◀─── 🎯 Semantic Search
```
1. **📄 PDF Document** — The source input file.
2. **🔍 Text Extraction** — Parsing and extracting raw text from the document.
3. **✂️ Chunking** — Segmenting the text into manageable, overlapping tokens.
4. **🧬 Gemini Embeddings** — Converting text chunks into high-dimensional vectors.
5. **🗄️ FAISS Vector Index** — Storing and indexing embeddings for fast similarity mapping.
6. **🎯 Semantic Search** — Matching the user's query against the indexed vectors.
7. **📥 Relevant Context Retrieval** — Fetching the most relevant text chunks.
8. **⚡ Gemini 2.5 Flash** — Passing the context and query to the language model.
9. **📝 Answer Generation** — Delivering the final synthesized response.

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


## Learning Objectives

This project demonstrates the core concepts behind modern Retrieval-Augmented Generation (RAG) systems:

* Embeddings
* Vector Databases
* Semantic Search
* Retrieval Pipelines
* Grounded LLM Responses
