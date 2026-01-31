# README

# RAG with Elasticsearch – Customer Support Assistant

This project is a fully local, end-to-end implementation of a **Retrieval-Augmented Generation (RAG)** pipeline
using **Elasticsearch as a vector database** and a **local open-source LLM**.

The goal is to demonstrate how modern semantic search and LLMs can be combined to build
a simple but realistic **customer support assistant**.

## 🧱 Architecture

User Question

↓

SentenceTransformer (embedding)

↓

Elasticsearch (vector search)

↓

Relevant FAQ chunks

↓

Prompt construction

↓

Local LLM (TinyLlama)

↓

Final Answer

---

## 🔎 Use case

A user asks a question to a customer support chatbot.  
The system:
1. Searches a FAQ knowledge base using semantic similarity
2. Retrieves the most relevant documents
3. Injects them into a prompt
4. Uses a local LLM to generate a grounded answer

This prevents hallucinations and makes the system explainable.

---

Elasticsearch is used as a **vector store**, not only as a keyword search engine.

---

## 🛠 Tech stack

- **Elasticsearch 8** (Docker)
- **Python**
- **Sentence Transformers** for embeddings
- **Hugging Face Transformers** for the LLM
- **TinyLlama 1.1B** (CPU-friendly model)

Everything runs **locally**, no API calls, no cloud.

---

## 🎯 Project goals

- Learn and practice **Elasticsearch vector search**
- Build a **RAG pipeline from scratch**
- Understand how embeddings, retrieval and LLMs interact
- Keep the system **simple, transparent and hackable**

No frameworks (LangChain, LlamaIndex…) are used on purpose.

---

## ⚠️ Known limitations

- The LLM is small and CPU-only → generation quality is limited
- The focus is on **retrieval**, not on fluent text generation
- No hybrid (BM25 + vector) search yet
- Minimal text preprocessing

Despite this, retrieval quality and traceability are realistic and production-relevant.

---

## 🚀 How to run

### 1. Clone the repo

```bash
git clone git@github.com:<your_username>/RAG_customer_support.git
cd RAG_customer_support
```
### 2. Start Elasticserach

```bash
cd docker
docker compose up -d
```
Check it works:
```bash
curl http://localhost:9200
```

### 3. Create Python environment
```bash
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
```

### 4. Create the index

```bash
python src/create_index.py
```

### 5. Ingest the FAQ

```bash
python src/ingest_faq.py
```

### 6. Test retrieval

```bash
python src/search.py
```

### 7. Run the full RAG pipeline
```bash
python src/rag_pipeline.py

```
Edit the question inside rag_pipeline.py to test different queries.

## What this project demonstrates

- Elasticsearch can act as a vector database
- Retrieval quality is independent of LLM size
- RAG systems are mainly about data and search
- Small models can still be useful when grounded with good retrieval

## Possible improvements

- Hybrid search (BM25 + vectors)
- Better multilingual LLM
- Reranking
- Web UI
- More advanced chunking

