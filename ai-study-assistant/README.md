# AI Study Assistant using Endee

## 📌 Overview
This project is an AI-powered chatbot that answers questions based on study material using vector embeddings and semantic search.

## 🚀 Features
- Semantic search
- Question answering system
- Lightweight RAG (Retrieval Augmented Generation)
- Fast response using embeddings

## 🧠 Tech Stack
- Python
- Flask
- Sentence Transformers
- Endee Vector Database

## ⚙️ How it Works
1. Text data is stored and converted into embeddings
2. Embeddings are stored using vector database principles
3. User query is converted into embedding
4. Most similar data is retrieved
5. Answer is returned

## 📁 Project Structure
ai-study-assistant/
├── app.py
├── data.txt
└── README.md

## ▶️ Setup Instructions
```bash
pip install sentence-transformers flask
python app.py
```

## 📡 API Endpoint
POST `/ask`
Example:
```json
{
  "query": "What is machine learning?"
}
```

## 🌐 Web Interface
This project includes a simple web-based chat interface where users can ask questions interactively.

Open in browser:
http://127.0.0.1:5000/

## 📄 PDF Upload Feature
Users can upload PDF documents, and the system will extract text and answer questions based on the uploaded content.
