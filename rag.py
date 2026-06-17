import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# ── LangChain imports ────────────────────────────────────
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Explicitly load environment variables from local .env
load_dotenv()


# ── 0. CUSTOM EMBEDDINGS WRAPPER ─────────────────────────
# LangChain's FAISS vector store needs an "Embeddings" object with two
# methods: embed_documents() for many texts, and embed_query() for one.
# We wrap our existing Gemini embedding call so LangChain can use it.
class GeminiEmbeddings(Embeddings):
    def __init__(self, client, model="gemini-embedding-2"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for i, text in enumerate(texts):
            print(f"    Embedding chunk {i+1}/{len(texts)}...")
            vectors.append(self._embed_one(text))
        return vectors

    def embed_query(self, text):
        return self._embed_one(text)

    def _embed_one(self, text):
        result = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return result.embeddings[0].values


# ── 1. LOAD (LangChain document loader) ──────────────────
def load_pdf(path):
    loader = PyMuPDFLoader(path)
    documents = loader.load()  # one LangChain "Document" per page, with metadata
    return documents


# ── 2. SPLIT INTO CHUNKS (LangChain text splitter) ───────
def split_into_chunks(documents, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ── 3 & 4. BUILD FAISS INDEX (LangChain FAISS vector store) ──
def build_faiss_index(embeddings, chunks):
    print("  Embedding chunks and building FAISS index (this may take a moment)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ── 5. FIND TOP CHUNKS BY MEANING ────────────────────────
def find_top_chunks(vector_store, question, top_n=3):
    results = vector_store.similarity_search_with_score(question, k=top_n)
    top_chunks = []
    for doc, score in results:
        top_chunks.append({
            "chunk": doc.page_content,
            "distance": float(score)
        })
    return top_chunks


# ── 6. ASK GEMINI ────────────────────────────────────────
def ask_gemini(client, question, top_chunks):
    context = "\n\n---\n\n".join([item["chunk"] for item in top_chunks])

    prompt = f"""You are answering only from the provided context.

If the answer is not present in the context,
say "I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except errors.ServerError as e:
        if "503" in str(e) or "high demand" in str(e).lower():
            return (
                "ERROR: Google's servers are currently overloaded with high traffic (503).\n"
                "Please wait 5-10 seconds and run your script again!"
            )
        raise e


# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment token is missing or blank!")

    client = genai.Client(api_key=api_key)
    embeddings = GeminiEmbeddings(client)

    print("Loading PDF...")
    documents = load_pdf("sample.pdf")
    print(f"  → {len(documents)} pages loaded")

    print("Splitting into chunks...")
    chunks = split_into_chunks(documents)
    print(f"  → {len(chunks)} chunks created")

    print("\nBuilding FAISS index...")
    vector_store = build_faiss_index(embeddings, chunks)
    print(f"  → Index built with {vector_store.index.ntotal} vectors")

    question = input("\nAsk a question: ")

    print("\nSearching by meaning...")
    top_chunks = find_top_chunks(vector_store, question, top_n=3)

    print(f"\n  Top {len(top_chunks)} chunks found:")
    for i, item in enumerate(top_chunks, 1):
        preview = item["chunk"][:80].replace("\n", " ")
        print(f"    Chunk {i} (distance {item['distance']:.2f}): {preview}...")

    print("\nAsking Gemini...")
    answer = ask_gemini(client, question, top_chunks)

    print("\n── ANSWER ──────────────────────────")
    print(answer)