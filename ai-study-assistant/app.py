from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pypdf import PdfReader

app = Flask(__name__, static_folder=".")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load initial data
with open("data.txt", "r") as f:
    documents = f.readlines()

# Create embeddings
doc_embeddings = model.encode(documents)


# 🔍 Search function (improved)
def search(query):
    query_embedding = model.encode([query])
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()

    # Get top 2 results for better answer
    top_indices = scores.argsort()[-2:][::-1]
    results = [documents[i] for i in top_indices]

    return " ".join(results)


# 📄 PDF loader
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text.split("\n")


# 🌐 Home route (UI)
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# 🤖 Ask API
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    answer = search(query)
    return jsonify({"answer": answer})


# 📄 Upload PDF API
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    file_path = file.filename
    file.save(file_path)

    global documents, doc_embeddings
    documents = load_pdf(file_path)
    doc_embeddings = model.encode(documents)

    return jsonify({"message": "PDF uploaded successfully"})


# ▶️ Run app
if __name__ == "__main__":
    app.run(debug=True)
