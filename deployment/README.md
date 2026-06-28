# 🤖 Phase 4 Hybrid RAG Chatbot

**RAG Internship Project**

A Retrieval-Augmented Generation (RAG) chatbot developed as part of the **RAG Internship**. This project combines **Hybrid Retrieval (Vector Search + BM25)** with **Google Gemini Embeddings**, **ChromaDB**, **Groq Llama 3.3 70B**, and **Streamlit** to answer questions from a custom knowledge base.

---

## 🚀 Features

* Hybrid Retrieval using **Vector Search + BM25**
* Semantic search using **Gemini Embedding Model**
* Vector storage using **ChromaDB**
* Fast answer generation using **Groq Llama 3.3 70B**
* Recursive Character Text Splitting
* Source Chunk Display for answer transparency
* Interactive Streamlit Chatbot
* Graceful handling of unknown questions
* Persistent Vector Database
* Clean and responsive user interface

---

## 🛠️ Technologies Used

| Technology                     | Purpose           |
| ------------------------------ | ----------------- |
| Python                         | Backend           |
| Streamlit                      | Web Application   |
| LangChain                      | RAG Framework     |
| ChromaDB                       | Vector Database   |
| Google Gemini Embeddings       | Text Embeddings   |
| Groq Llama 3.3 70B             | Answer Generation |
| BM25                           | Keyword Retrieval |
| RecursiveCharacterTextSplitter | Document Chunking |

---

## 📂 Project Structure

```text
RAG_Internship/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .env
│
├── wiki_docs/
│
├── phase4_chroma_db/
│
├── outputs/
│
├── notebooks/
│
├── assets/
│
└── rag_env/
```

---

## ⚙️ How It Works

1. Load the knowledge base.
2. Split documents into meaningful chunks.
3. Generate embeddings using Gemini.
4. Store embeddings inside ChromaDB.
5. Retrieve relevant chunks using:

   * Vector Similarity Search
   * BM25 Keyword Search
6. Merge the retrieved results.
7. Generate the final answer using Groq Llama 3.3 70B.
8. Display the retrieved source chunks along with the answer.

---

## 🧠 Hybrid Retrieval Pipeline

```text
User Question
       │
       ▼
Hybrid Retrieval
 ├── ChromaDB Vector Search
 └── BM25 Keyword Search
       │
       ▼
Merged Relevant Chunks
       │
       ▼
Groq Llama 3.3 70B
       │
       ▼
Final Answer + Source Chunks
```

---

## 📚 Knowledge Base

The chatbot is built using curated Wikipedia documents covering topics such as:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Neural Networks
* Computer Vision
* Natural Language Processing
* Reinforcement Learning
* Large Language Models
* Generative AI
* Retrieval-Augmented Generation

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/shifasulthana/rag-internship-2026.git
cd rag-internship-2026
```

Create a virtual environment

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```text
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Chatbot

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 💬 Example Questions

* What is Machine Learning?
* Explain Deep Learning.
* What are Neural Networks?
* What is Retrieval-Augmented Generation?
* Explain Large Language Models.
* What is Reinforcement Learning?

---

## 📷 Screenshots

### Home Page

![Home](assets/home.png)

### Chatbot Response

![Response](assets/response.png)

### Source Chunks

![Sources](assets/sources.png)

---

## 📈 Improvements Implemented (Phase 4)

* Hybrid Retrieval (Vector Search + BM25)
* Improved Recursive Chunking
* Persistent ChromaDB Storage
* Source Chunk Display
* Streamlit Deployment
* Better Error Handling

---

## 🔮 Future Improvements

* Cross Encoder Re-ranking
* Conversation Memory
* PDF Upload Support
* Multi-document Knowledge Base
* Chat History Export
* Authentication
* Cloud Deployment

---

## 👩‍💻 Author

**Shifa Sulthana**

RAG Internship Project

GitHub: https://github.com/shifasulthana/rag-internship-2026

---

## 📄 License

This project was developed for educational and internship purposes.
