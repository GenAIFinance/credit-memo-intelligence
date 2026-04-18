# Credit Risk Theme & Workflow Intelligence Platform

A system that turns unstructured credit research and portfolio-monitoring text into a searchable taxonomy of risk themes, issuer events, recommended actions, and historical analogs.

## Stack
- **LLM**: Azure OpenAI GPT-4o (exploration + generation)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, open source)
- **Orchestration**: LangChain
- **Vector store**: ChromaDB
- **Data layer**: Pandas + DuckDB
- **UI**: Streamlit

## Setup
```bash
cp .env.example .env       # fill in Azure OpenAI credentials
pip install -r requirements.txt
python -m src.generation.generate_corpus --target 100   # quick test
```

## Pipeline phases
| Phase | Script | Description |
|---|---|---|
| 1 | `src/generation/generate_corpus.py` | Synthetic corpus via Azure OpenAI |
| 2 | `src/preprocessing/normalize.py` | Clean, chunk (LangChain), DuckDB |
| 3 | `src/exploration/llm_explorer.py` | Taxonomy + slot extraction via GPT-4o |
| 4 | `src/embeddings/embed_cluster.py` | BGE embeddings + KMeans + UMAP |
| 5 | `src/retrieval/retriever.py` | LangChain VectorStoreRetriever |
| 6 | `app/app.py` | Streamlit UI |
