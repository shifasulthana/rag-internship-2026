"""
Phase 4 Improved RAG Pipeline
PressGaney Internship - Shifa Sulthana O

Hybrid Search: Vector (ChromaDB + Gemini) + BM25 (rank-bm25)
Embeds in batches of 20 with 65s sleep between batches to stay
within Gemini free-tier rate limits (100 requests/minute).
"""

# ============================================================
# SQLite Fix for Ubuntu 20.04 — MUST be first import
# ============================================================
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# ============================================================
# Standard Library
# ============================================================
import os
import re
import time
import shutil
from pathlib import Path
from typing import List

# ============================================================
# Third-Party
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
from rank_bm25 import BM25Okapi

# ============================================================
# Load Environment Variables
# ============================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ============================================================

# Configuration — matches Hybrid_Chunking.ipynb exactly

# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL       = "llama-3.3-70b-versatile"

CHROMA_PATH     = BASE_DIR / "phase4_chroma_db"
WIKI_DIR        = BASE_DIR / "wiki_docs"

COLLECTION_NAME = "wiki_rag_phase4"

CHUNK_SIZE      = 1800
CHUNK_OVERLAP   = 200

BATCH_SIZE      = 20
BATCH_SLEEP     = 65

# ============================================================
# Initialize Models
# ============================================================
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# Load or Create ChromaDB
# ============================================================
def load_or_create_vectordb() -> Chroma:
    """
    Loads an existing Phase 4 ChromaDB if valid.
    Otherwise creates a new one from wiki_docs.

    Features:
    ✓ Detects corrupted/empty database
    ✓ Rebuilds automatically
    ✓ Embeds in batches
    ✓ Retries only on quota errors (429)
    ✓ Prints detailed progress
    """

    # --------------------------------------------------
    # Try loading existing database
    # --------------------------------------------------
    if Path(CHROMA_PATH).exists():

        print("\nChecking existing ChromaDB...")

        try:

            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )

            count = db._collection.count()

            if count > 0:
                print(f"✓ Existing ChromaDB loaded ({count} chunks).")
                return db

            print("Existing database is empty.")
            shutil.rmtree(CHROMA_PATH)

        except Exception as e:

            print(f"Existing database is unusable:\n{e}")

            shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    # --------------------------------------------------
    # Build new database
    # --------------------------------------------------

    print("\nCreating Phase 4 ChromaDB from scratch...\n")

    if not WIKI_DIR.exists():

        raise FileNotFoundError(
            f"wiki_docs folder not found:\n{WIKI_DIR.resolve()}"
        )

    loader = DirectoryLoader(
        str(WIKI_DIR),
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:

        raise ValueError(
            "No text documents found."
        )

    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    total = len(chunks)

    print(f"Created {total} chunks.\n")

    db = None

    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_index in range(total_batches):

        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)

        batch = chunks[start:end]

        print("=" * 60)
        print(f"Batch {batch_index + 1}/{total_batches}")
        print(f"Chunks : {start + 1} - {end}")
        print("=" * 60)

        success = False

        while not success:

            try:

                if db is None:

                    db = Chroma.from_documents(

                        documents=batch,

                        embedding=embeddings,

                        persist_directory=CHROMA_PATH,

                        collection_name=COLLECTION_NAME

                    )

                else:

                    db.add_documents(batch)

                print("✓ Batch completed.\n")

                success = True

            except Exception as e:

                error = str(e)

                if "429" in error or "quota" in error.lower():

                    print("\nGemini quota reached.")
                    print(f"Waiting {BATCH_SLEEP} seconds...\n")

                    time.sleep(BATCH_SLEEP)

                else:

                    raise RuntimeError(
                        f"\nBatch {batch_index + 1} failed.\n\n{e}"
                    )

        if batch_index != total_batches - 1:

            print(
                f"Sleeping {BATCH_SLEEP} seconds before next batch...\n"
            )

            time.sleep(BATCH_SLEEP)

    print("\n✓ Phase 4 ChromaDB created successfully.")

    print(f"Stored {db._collection.count()} chunks.\n")

    return db


# ============================================================
# Initialize DB at module load time
# ============================================================
try:
    vectordb = load_or_create_vectordb()
except Exception as e:
    raise RuntimeError(f"Vector database initialization failed:\n{e}")

# ============================================================
# Build BM25 index from the same chunks in ChromaDB
# ============================================================
_raw        = vectordb.get()
chunk_texts: List[str] = _raw["documents"]

tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index built on {len(chunk_texts)} chunks.")

# ============================================================
# Search Functions — exact logic from Hybrid_Chunking.ipynb
# ============================================================
def vector_search(query: str, k: int = 3) -> list:
    """Semantic search using ChromaDB + Gemini embeddings."""
    try:
        return vectordb.similarity_search(query, k=k)
    except Exception as e:
        print(f"Vector Search Error: {e}")
        return []


def bm25_search(query: str, k: int = 2) -> list:
    """Keyword search using BM25Okapi.
    Returns list of (text, score) tuples."""
    try:
        tokens = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(tokens)
        ranked = sorted(
            zip(chunk_texts, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:k]
    except Exception as e:
        print(f"BM25 Search Error: {e}")
        return []


def hybrid_search(query: str, vector_k: int = 3, bm25_k: int = 2) -> list:
    """Combines vector + BM25 results.
    Vector results are prioritised; BM25 adds unique chunks only."""
    vector_docs = vector_search(query, vector_k)
    bm25_docs   = bm25_search(query, bm25_k)

    final_docs = []
    seen       = set()

    # Prioritise vector results (LangChain Document objects)
    for doc in vector_docs:
        text = doc.page_content
        if text not in seen:
            final_docs.append(doc)
            seen.add(text)

    # Add BM25 results (raw strings)
    for text, score in bm25_docs:
        if text not in seen:
            final_docs.append(text)
            seen.add(text)

    return final_docs


# ============================================================
# Main Function — called by app.py
# ============================================================
def ask_question(query: str) -> dict:
    """
    Runs the full hybrid RAG pipeline.

    Returns a dict:
        answer   : str  — the LLM answer
        sources  : list — chunk texts shown in the UI expander
        num_docs : int  — number of chunks retrieved
    """
    try:
        docs = hybrid_search(query)

        if not docs:
            return {
                "answer":   "I could not find that information in the documents.",
                "sources":  [],
                "num_docs": 0
            }

        # Build context — handles both Document objects and raw strings
        source_texts = []
        for doc in docs:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            source_texts.append(text)

        context = "\n\n".join(source_texts)

        prompt = f"""You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:
'I could not find that information in the documents.'

Context:
{context}

Question:
{query}"""

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer":   response.choices[0].message.content,
            "sources":  source_texts,
            "num_docs": len(docs)
        }

    except Exception as e:
        return {
            "answer":   f"Error: {e}",
            "sources":  [],
            "num_docs": 0
        }