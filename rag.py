import fitz
import numpy as np
import faiss
import os
import time
from google import genai
from google.genai import errors
from dotenv import load_dotenv

# Explicitly load environment variables from local .env
load_dotenv()

# ── 1. LOAD ──────────────────────────────────────────────
def load_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ── 2. SPLIT INTO CHUNKS ─────────────────────────────────
def split_into_chunks(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# ── 3. GET EMBEDDING FOR ONE PIECE OF TEXT ───────────────
def get_embedding(client, text):
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    vector = result.embeddings[0].values
    return np.array(vector, dtype="float32")

# ── 4. BUILD FAISS INDEX FROM ALL CHUNKS ─────────────────
def build_faiss_index(client, chunks):
    print("  Embedding chunks (this may take a moment)...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"    Embedding chunk {i+1}/{len(chunks)}...")
        vec = get_embedding(client, chunk)
        embeddings.append(vec)

    matrix = np.stack(embeddings)
    dimension = matrix.shape[1]           
    index = faiss.IndexFlatL2(dimension)  
    index.add(matrix)                     

    return index, embeddings

# ── 5. FIND TOP CHUNKS BY MEANING ────────────────────────
def find_top_chunks(client, question, chunks, index, top_n=3):
    question_vec = get_embedding(client, question)
    question_vec = question_vec.reshape(1, -1)

    # FIXED: Ensure top_n is never larger than our actual total chunk count
    actual_top_n = min(top_n, len(chunks))

    distances, indices = index.search(question_vec, actual_top_n)

    top_chunks = []
    for i, idx in enumerate(indices[0]):
        top_chunks.append({
            "chunk": chunks[idx],
            "distance": float(distances[0][i])   
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

    # FIXED: Wrapped in a try-except block to handle 503 high demand spikes cleanly
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

    print("Loading PDF...")
    text = load_pdf("sample.pdf")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text, chunk_size=300)
    print(f"  → {len(chunks)} chunks created")

    print("\nBuilding FAISS index...")
    index, embeddings = build_faiss_index(client, chunks)
    print(f"  → Index built with {index.ntotal} vectors")

    question = input("\nAsk a question: ")

    print("\nSearching by meaning...")
    top_chunks = find_top_chunks(client, question, chunks, index, top_n=3)

    print(f"\n  Top {len(top_chunks)} chunks found:")
    for i, item in enumerate(top_chunks, 1):
        preview = item["chunk"][:80].replace("\n", " ")
        print(f"    Chunk {i} (distance {item['distance']:.2f}): {preview}...")

    print("\nAsking Gemini...")
    answer = ask_gemini(client, question, top_chunks)

    print("\n── ANSWER ──────────────────────────")
    print(answer)