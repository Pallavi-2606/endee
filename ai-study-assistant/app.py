from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("ai-study-assistant/data.txt", "r") as f:
    documents = f.readlines()

# Create embeddings
doc_embeddings = model.encode(documents)

# Search function
def search(query):
    query_embedding = model.encode([query])
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    return documents[np.argmax(scores)]

# API
@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    answer = search(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
