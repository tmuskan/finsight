# FinSight

A financial research AI assistant that combines fine-tuned BERT models,
retrieval-augmented generation (RAG), and a multi-agent system over SEC
filings, financial news, and earnings call transcripts.

## Status
In development — Phase 0: Setup

## Architecture
- Code is developed locally and pushed to GitHub.
- All ML training and inference runs on Kaggle Notebooks (free GPU).
- Trained models and processed datasets live on the Hugging Face Hub,
  which acts as the bridge between development and runtime.

## Stack
- Models: BERT (sentiment, NER), Llama 3.1 8B (QLoRA fine-tuned)
- Embeddings: sentence-transformers
- Vector DB: ChromaDB
- RAG framework: LangChain
- Compute: Kaggle Notebooks
- Storage: Hugging Face Hub
- Serving: FastAPI + Streamlit (final phase)