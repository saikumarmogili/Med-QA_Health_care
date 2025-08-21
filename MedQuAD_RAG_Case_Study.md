# MedQuAD QA — Retrieval-Augmented Medical Q&A (Extractive)

**Summary.** A retrieval-augmented QA system for consumer health questions. It retrieves evidence from a curated corpus, re-ranks for precision, returns short **extractive** answers with **citations**, and says **“I’m not sure”** when evidence is weak. No web browsing; only the indexed corpus is used.

## Problem
Patients and support teams need trustworthy, concise explanations with sources. Pure generation can hallucinate; keyword search misses synonyms.

## Data
- Source: curated consumer-health Q&A (e.g., MedQuAD/NIH pages).
- Preprocessing: strip HTML → sentence chunking (2–3 sentences, small overlap).
- Indexing: FAISS (dense cosine via normalized dot) + BM25 (lexical).

## Architecture (RAG-Extractive)
1) Safety gates (emergency/dosing)  
2) Stage-1 retrieval: BM25 ⊕ bi-encoder (hybrid) → top-N candidates (100–200)  
3) Stage-2 re-ranking: cross-encoder on (query, passage) → top-K (5)  
4) Answer: extract best-supported sentence(s) + citations; low-confidence → “I’m not sure”  
5) UI/API: Streamlit/Gradio + FastAPI `/ask`

## Models
- Retriever: `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical IR)
- Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Optional fine-tunes (1–2 epochs). Confidence via sigmoid or isotonic calibration.

## Evaluation
- **Recall@50/100** (retrieval safety), **MRR@10**, **nDCG@10** (rank quality)
- Ops: stage latencies, unsure-rate
- Goal: high Recall@100 (≥0.9), improved MRR@10 with re-ranking

## Example Results (fill from your runs)
| Approach | Recall@100 | MRR@10 | nDCG@10 | P95 Latency | Notes |
|---|---:|---:|---:|---:|---|
| BM25 only |  |  |  |  |  |
| Dense (bi-encoder) |  |  |  |  |  |
| Hybrid (BM25 ⊕ Dense) |  |  |  |  |  |
| Hybrid + Re-ranker |  |  |  |  |  |

## Safety & Grounding
Educational use only; no dosing/diagnosis. Answers are **extractive** from the corpus with citations. Uncertainty threshold triggers “I’m not sure”.

## Notable Decisions
Hybrid retrieval for recall; cross-encoder for early precision; extractive default to minimize hallucination.

## Future Work
Confidence calibration; synonym/acronym expansion; hard-negative mining; evidence highlighting; dashboard.
