import os
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv


load_dotenv("api.env")

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")
ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ask My Documents</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 60px;
            margin: 0;
        }
        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            width: 100%;
            max-width: 700px;
        }
        h2 {
            margin-top: 0;
            color: #333;
            font-size: 28px;
        }
        form {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        input[type=text], input:not([type]) {
            flex: 1;
            padding: 14px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type=text]:focus, input:not([type]):focus {
            border-color: #764ba2;
        }
        button {
            padding: 14px 28px;
            background: #764ba2;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #5f3d85;
        }
        .answer-box {
            margin-top: 20px;
            padding: 20px;
            background: #f8f7fc;
            border-left: 4px solid #764ba2;
            border-radius: 8px;
            line-height: 1.6;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>Ask My Documents 🤖</h2>
        <form method="POST">
          <input name="question" placeholder="Ask something...">
          <button type="submit">Ask</button>
        </form>
        {% if answer %}
        <div class="answer-box"><b>Answer:</b> {{ answer }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        q_embedding = model.encode(question).tolist()
        results = collection.query(query_embeddings=[q_embedding], n_results=3)
        context = "\n\n".join(results["documents"][0])

        response = ai.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Answer the question using ONLY this context:\n\n{context}\n\nQuestion: {question}"
        )
        answer = response.text
    return render_template_string(PAGE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)