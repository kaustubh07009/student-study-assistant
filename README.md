# 🎓 AI Student Study Assistant

An AI-powered study assistant that allows students to upload study documents, ask questions, and generate quizzes using Google Gemini and FAISS-based semantic search.

## 🚀 Features

- 📄 Upload PDF study materials
- 🤖 AI-powered chatbot
- 🧭 Intelligent question routing
- 📚 Document-based Question & Answer
- 🧠 AI-generated quizzes
- 📝 Automatic quiz scoring
- ✂️ Document chunking
- 🔎 Semantic search using FAISS
- 🔢 Gemini-powered document embeddings
- 💬 General study assistance

## 🏗️ Architecture

```text
Student
   ↓
Streamlit Interface
   ↓
AI Router
   ├── Chat Agent
   │      ↓
   │    Gemini
   │
   ├── PDF Agent
   │      ↓
   │   Gemini Embeddings
   │      ↓
   │    FAISS Search
   │      ↓
   │   Relevant Chunks
   │      ↓
   │    Gemini
   │
   └── Quiz Agent
          ↓
      FAISS Search
          ↓
      Relevant Chunks
          ↓
        Gemini