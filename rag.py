import fitz
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# ── 1. LOAD ──────────────────────────────────────────────
def load_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ── 2. SPLIT INTO CHUNKS ─────────────────────────────────
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# ── 3. FIND TOP 3 CHUNKS ─────────────────────────────────
def find_top_chunks(question, chunks, top_n=3):            # keyword overlap search
    question_words = set(question.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words & chunk_words)
        scored.append((overlap, chunk))                    # store score WITH the chunk

    scored.sort(key=lambda x: x[0], reverse=True)         # sort highest score first
    top_chunks = [chunk for score, chunk in scored[:top_n]] # take only top 3
    return top_chunks

# ── 4. ASK GEMINI ────────────────────────────────────────
def ask_gemini(question, top_chunks):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    context = "\n\n---\n\n".join(top_chunks)               # join 3 chunks with a divider

    prompt = f"""You are answering only from the provided context.

If the answer is not present in the context,
say "I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf("sample.pdf")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text, chunk_size=500)
    print(f"  → {len(chunks)} chunks created")

    question = input("\nAsk a question: ")

    print("\nSearching for relevant chunks...")
    top_chunks = find_top_chunks(question, chunks, top_n=3)

    print(f"  → Top {len(top_chunks)} chunks found:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"    Chunk {i}: {chunk[:80]}...")  # show first 80 characters as preview

    print("\nAsking Gemini...")
    answer = ask_gemini(question, top_chunks)

    print("\n── ANSWER ──────────────────────────")
    print(answer)