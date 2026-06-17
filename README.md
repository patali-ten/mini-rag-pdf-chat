# PDF Question Answering RAG
A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a PDF document using semantic search and Google's Gemini models, built with LangChain.

## Features
* Extract text from PDF documents using LangChain's PyMuPDF document loader
* Split documents into manageable, overlapping chunks using LangChain's RecursiveCharacterTextSplitter
* Generate embeddings using Gemini Embedding Model
* Store and search embeddings using LangChain's FAISS vector store
* Retrieve the most relevant chunks using semantic similarity
* Generate grounded answers with Gemini 2.5 Flash
* Reduce hallucinations by restricting answers to retrieved context
* Secure API key management using environment variables
  
## Tech Stack
* Python
* LangChain (Document Loaders, Text Splitters, Vector Stores)
* Gemini 2.5 Flash
* Gemini Embedding Model
* FAISS
* PyMuPDF
* NumPy
* python-dotenv
  
## Architecture
```text
📄 PDF Document ➔ 🔍 LangChain Document Loader ➔ ✂️ LangChain Text Splitter ➔ 🧬 Gemini Embeddings ➔ 🗄️ LangChain FAISS Vector Store
                                                                                                                          │
📝 Answer Generation 💾 💡◀─── ⚡ Gemini 2.5 Flash ◀─── 📥 Context Retrieval ◀─── 🎯 Semantic Search

```

1. **📄 PDF Document** — The source input file.
2. **🔍 LangChain Document Loader** — Using `PyMuPDFLoader` to parse the PDF into page-level `Document` objects.
3. **✂️ LangChain Text Splitter** — Using `RecursiveCharacterTextSplitter` to segment the text into manageable, overlapping chunks.
4. **🧬 Gemini Embeddings** — Converting text chunks into high-dimensional vectors.
5. **🗄️ LangChain FAISS Vector Store** — Storing and indexing embeddings for fast similarity mapping.
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
1. Load a PDF document using a LangChain document loader
2. Split the document into chunks using a LangChain text splitter
3. Generate embeddings for each chunk
4. Build a LangChain FAISS vector store
5. Ask a question
6. Retrieve the most relevant chunks via similarity search
7. Generate an answer using Gemini

## Learning Objectives
This project demonstrates the core concepts behind modern Retrieval-Augmented Generation (RAG) systems:
* Embeddings
* Vector Databases
* Semantic Search
* Retrieval Pipelines
* Grounded LLM Responses
* Using LangChain's document loaders, text splitters, and vector store abstractions
