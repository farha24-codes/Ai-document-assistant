import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

docs_folder = "docs"
chunk_id = 0
for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        print(f"Reading {filename}...")
        text = read_pdf(os.path.join(docs_folder, filename))
        chunks = chunk_text(text)
        for chunk in chunks:
            if chunk.strip():
                embedding = model.encode(chunk).tolist()
                collection.add(
                    ids=[str(chunk_id)],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": filename}]
                )
                chunk_id += 1

print(f"Done! Stored {chunk_id} chunks in the memory box.")