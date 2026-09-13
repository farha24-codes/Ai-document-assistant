# AI Document Assistant (RAG)

A retrieval-augmented Q&A tool: upload PDFs, ask questions, get answers grounded in the actual document content.

## How it works
1. PDFs are split into chunks and converted into embeddings (sentence-transformers)
2. Chunks are stored in a ChromaDB vector database
3. On a question, the most relevant chunks are retrieved via semantic search
4. Gemini generates an answer using only that retrieved context (RAG pattern)

## Tech stack
Python, ChromaDB, sentence-transformers, Google Gemini API (gemini-3-flash-preview), Flask

## How to Run
 1. Clone the repository using: git clone https://github.com/farha24-codes/Ai-document-assistant.git then cd Ai-document-assistant
 2. Create and activate a virtual environment using: python -m venv venv then venv\Scripts\activate
 3. Install dependencies using: pip install -r Requirements.txt
 4. Add your Gemini API key by creating a file named api.env in the project root and adding: GEMINI_API_KEY=your_api_key_here (Get a free key at Google AI Studio :https://aistudio.google.com/apikey)
 5. Add your PDFs by placing any PDF documents you want to query inside the docs folder.
 6. Run ingestion using: python ingest.py which processes your PDFs and stores them in a local vector database.
 7. Run the app using: python app.py then open http://127.0.0.1:5000 in your browser and start asking questions.
