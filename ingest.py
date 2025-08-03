# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load API key from .env if available
load_dotenv()

# --- Config ---
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- 1. Load all documents ---
def load_documents():
    documents = []

    # Load PDF
    pdf_path = os.path.join(DATA_DIR, "project_docs.pdf")
    if os.path.exists(pdf_path):
        pdf_loader = PyPDFLoader(pdf_path)
        documents.extend(pdf_loader.load())
        print(f"✅ Loaded PDF: {pdf_path}")
    else:
        print(f"⚠️ PDF not found: {pdf_path}")

    # Load SQL files
    for fname in ["schema.sql", "queries.sql"]:
        sql_path = os.path.join(DATA_DIR, fname)
        if os.path.exists(sql_path):
            loader = TextLoader(sql_path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"✅ Loaded SQL: {sql_path}")
        else:
            print(f"⚠️ SQL not found: {sql_path}")

    return documents

# --- 2. Chunk documents ---
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# --- 3. Create & Save Vectorstore ---
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()  # Uses OPENAI_API_KEY env var
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ Vectorstore saved to '{VECTORSTORE_DIR}'")

# --- Main ---
if __name__ == "__main__":
    try:
        docs = load_documents()
        chunks = split_documents(docs)
        build_vectorstore(chunks)
    except Exception as e:
        print(f"❌ Error: {e}")
