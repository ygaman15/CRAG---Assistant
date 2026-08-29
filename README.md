
Built as a **LangGraph** state machine (`retrieve → grade_documents → [web_search] → generate`).

## Key Features

- **Hybrid Retrieval**: Combines dense semantic retrieval (Chroma + `all-MiniLM-L6-v2` embeddings) with sparse keyword retrieval (BM25), then deduplicates and merges candidates.
- **Cross-Encoder Reranking**: Re-scores merged candidates using `BAAI/bge-reranker-base` for higher-precision top-k selection.
- **LLM-Based Relevance Grading**: A fast grader model (Groq `gpt-oss-20b`) evaluates whether each retrieved passage is actually relevant to the question, filtering out noise.
- **Corrective Web Fallback**: If no locally retrieved documents pass grading, the system automatically queries the web via Tavily Search — and grades those results too, closing the corrective loop.
- **Grounded Generation**: A stronger generation model (Groq `gpt-oss-120b`) answers strictly from graded, relevant context, with a hard guard against hallucination when no evidence survives grading.

## Tech Stack

- **Orchestration**: LangGraph
- **Retrieval**: LangChain, Chroma (vector store), BM25Retriever, HuggingFace embeddings
- **Reranking**: HuggingFace Cross-Encoder (`BAAI/bge-reranker-base`)
- **LLMs**: Groq (`gpt-oss-20b` for grading, `gpt-oss-120b` for generation)
- **Web Search Fallback**: Tavily Search API
- **Language**: Python

## Project Structure

- `rag_engine.py` — Core `HybridCRAGPipeline` class: retrieval, reranking, grading, web fallback, and generation logic
- `app.py` — Application entry point / UI layer
- `requirements.txt` — Project dependencies

## Setup

1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Set required environment variables:
```bash
   export GROQ_API_KEY=your_groq_api_key
   export TAVILY_API_KEY=your_tavily_api_key
```
4. Run the app:
```bash
   python app.py
```

## How It Works

1. **Retrieve**: Fetch top candidates via both dense and sparse retrieval, then rerank with a cross-encoder.
2. **Grade**: An LLM grader labels each passage as relevant (YES) or not (NO).
3. **Correct (if needed)**: If zero passages are relevant, search the web and grade those results too.
4. **Generate**: Answer strictly from the surviving relevant context, citing sources; if no evidence remains, the system explicitly states it doesn't know rather than hallucinating.
