# Credit Memo Intelligence Platform

> **MVP** — built to demonstrate LLM-first data exploration, semantic clustering,
> taxonomy mapping, and LangChain retrieval/chunking modules on unstructured
> financial text. Scope is intentionally limited to what the pipeline does well
> at this stage; known limitations and roadmap items are listed below.

---

## What it does

Ingests synthetic credit memos for leveraged loan issuers and turns them into a
searchable intelligence layer. Given a new memo or a free-text query, the system
returns semantically similar historical memos, their taxonomy labels, and the
structured evidence that justifies each label.

The key capability is **combined structured + semantic search**:

```python
# "Find memos like this one, but only where leverage > 7x and collateral = first lien"
retriever.query(
    text="covenant headroom deteriorating, sponsor unlikely to support",
    filters={"net_leverage": {"$gt": 7.0}, "collateral": "first_lien"}
)
```

---

## Skills demonstrated

| Skill (from job description) | Implementation |
|---|---|
| LLM-first exploratory analysis | GPT-4o labels 200-doc sample; taxonomy discovered from data, not imposed top-down |
| Rapid semantic clustering | BGE-small embeddings → KMeans → UMAP 2D cluster map |
| Scaling prompt prototypes → embedding pipelines | Explicit two-stage: GPT-4o prototype (200 docs) → BGE pipeline (full corpus) |
| Taxonomy mapping + functional decomposition | Auto-extracts four slots per memo: trigger · risk signal · analyst stance · forward view |
| Mixed data processing (text + tables) | Section-aware chunker + GPT-4o table serialization |
| LangChain retrieval + chunking modules | RecursiveCharacterTextSplitter and VectorStoreRetriever as named, testable modules |

---

## Pipeline

```
Phase 1  generate_corpus.py      Synthetic credit memos via Azure OpenAI GPT-4o
Phase 2  normalize.py            Section-aware chunking via LangChain TextSplitter
Phase 3  llm_explorer.py         Taxonomy discovery + functional decomposition (200-doc sample)
Phase 4  embed_cluster.py        BGE-small embeddings + KMeans clustering + UMAP
Phase 5  retriever.py            LangChain VectorStoreRetriever over ChromaDB
Phase 6  app.py                  Streamlit demo UI
```

---

## Demo UI — four pages

### 1. Taxonomy explorer
Shows the taxonomy discovered by GPT-4o from the memo sample — risk themes,
action types, and forward views with frequency distributions. Filter the corpus
by any taxonomy label and read the evidence slots that justified it.

*Demonstrates: LLM-first exploratory analysis, taxonomy mapping.*

### 2. Cluster map
Interactive UMAP scatter plot — each point is a memo chunk, colored by cluster.
Click any point to read the memo. Toggle between coloring by cluster, by risk
theme, or by collateral type.

*Demonstrates: rapid semantic clustering, scaling prompt prototype to embedding pipeline.*

### 3. Similar memo retrieval
Paste a memo excerpt or select an issuer. Returns top-k semantically similar
historical memos with taxonomy labels, evidence slots, and key metrics
(net leverage, collateral, ESG score). Supports metadata filters.

*Demonstrates: LangChain retrieval module, combined structured + semantic search.*

### 4. Functional decomposition view
Select any memo and see its four auto-extracted slots alongside the source text:
trigger, risk signal, analyst stance, forward view.

*Demonstrates: functional decomposition automation.*

---

## Stack

| Layer | Tool | Reason |
|---|---|---|
| LLM | Azure OpenAI GPT-4o | Generation, taxonomy discovery, table serialization |
| Embeddings | BAAI/bge-small-en-v1.5 | Open source, strong on financial text, runs locally |
| Orchestration | LangChain | Chunking and retrieval modules |
| Vector store | ChromaDB | Free, local, metadata filtering, scales to hosted |
| Data layer | Pandas + DuckDB | Fast in-process analytics on structured metadata |
| UI | Streamlit | Fast to build, screen-recordable, deployable to Streamlit Cloud |

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/credit-memo-intelligence.git
cd credit-memo-intelligence
pip install -r requirements.txt

# 2. Configure Azure OpenAI
cp .env.example .env
# fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT

# 3. Run pipeline (full)
python -m src.generation.generate_corpus
python -m src.preprocessing.normalize
python -m src.exploration.llm_explorer
python -m src.embeddings.embed_cluster
python -m src.retrieval.retriever --build
streamlit run app/app.py

# Quick test run (100 docs)
python -m src.generation.generate_corpus --target 100
```

---

## MVP scope and known limitations

| Feature | Status | Notes |
|---|---|---|
| Competitive landscape image embeddings | Roadmap | Requires GPT-4o Vision |
| Mandatory checklist structured storage | Roadmap | Binary fields — metadata extension |
| Full 17-subsection company overview | Roadmap | MVP uses 5 key subsections |
| Real credit memo ingestion (PDF/DOCX) | Roadmap | MVP uses synthetic corpus |
| Hosted deployment | Roadmap | One-step extension via Streamlit Cloud |

---

## Project structure

```
credit-memo-intelligence/
├── config/config.yaml
├── prompts/
│   ├── generation_prompt.txt
│   ├── taxonomy_prompt.txt
│   └── table_serialization_prompt.txt
├── src/
│   ├── generation/generate_corpus.py
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   └── table_serializer.py
│   ├── exploration/llm_explorer.py
│   ├── embeddings/embed_cluster.py
│   └── retrieval/retriever.py
├── app/app.py
├── data/                              # gitignored
├── .env.example
├── requirements.txt
└── README.md
```
