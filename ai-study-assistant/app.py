from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__, static_folder=".")

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("ai-study-assistant/data.txt", "r") as f:
    documents = f.readlines()

doc_embeddings = model.encode(documents)

def search(query):
    query_embedding = model.encode([query])
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    return documents[np.argmax(scores)]

@app.route("/")
def home():
    return send_from_directory("ai-study-assistant", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    answer = search(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
