# PDF Question Answering RAG

A simple Retrieval-Augmented Generation (RAG) system built with Python and Gemini API.

## Features

* Extract text from PDF documents
* Split documents into chunks
* Retrieve relevant chunks based on user questions
* Use Gemini 2.5 Flash to generate answers
* Restrict answers to document content to reduce hallucinations

## Tech Stack

* Python
* PyMuPDF
* Gemini API
* python-dotenv

## How It Works

1. Load PDF document
2. Split text into chunks
3. Find relevant chunks
4. Send retrieved context and question to Gemini
5. Return grounded answer

## Future Improvements

* Embedding-based retrieval
* Vector database integration
* Web interface
* Multi-document support
