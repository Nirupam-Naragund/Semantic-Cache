# Semantic Cache with Redis + Gemini

This project demonstrates how to build a **semantic cache** using **Redis**, **vector embeddings**, and **Google Gemini** to avoid repeated LLM calls for semantically similar questions.

Instead of matching exact strings, the cache reuses answers based on **meaning**.



## 🚀 How It Works

1. User asks a question
2. Question is converted into an embedding (vector)
3. Redis searches for semantically similar past questions
4. If similarity distance is below a threshold → **CACHE HIT**
5. Otherwise → **CACHE MISS**, Gemini is called and the result is cached

## Steps to run

```bash
pip install -r requirements.txt
```

```bash
python main.py
```
