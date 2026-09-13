# AI Document Assistant (RAG)

A retrieval-augmented Q&A tool: upload PDFs, ask questions, get answers grounded in the actual document content.

## How it works
1. PDFs are split into chunks and converted into embeddings (sentence-transformers)
2. Chunks are stored in a ChromaDB vector database
3. On a question, the most relevant chunks are retrieved via semantic search
4. Gemini generates an answer using only that retrieved context (RAG pattern)

## Tech stack
Python, ChromaDB, sentence-transformers, Google Gemini API (gemini-3-flash-preview), Flask

## Run locally
1. `pip install -r requirements.txt`
2. Add your `GEMINI_API_KEY` to a `.env` file
3. Put PDFs in `/docs`
4. `python ingest.py`
5. `python app.py`
