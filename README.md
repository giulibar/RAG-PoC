# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, ask questions in natural language, and receive contextually grounded answers — built with Python on Google Colab.

---

## Overview

This project implements a full RAG pipeline from scratch. It chunks documents, generates semantic embeddings, stores them in a vector database, and uses a large language model to generate accurate, document-grounded responses.

**Stack:**
- **Embeddings:** Universal Sentence Encoder (USE) via TensorFlow Hub
- **Vector Store:** Elasticsearch
- **LLM:** LLaMA 3.3 70B Instruct Turbo (via Together AI)
- **Interface:** Gradio
- **Environment:** Google Colab

---

## Pipeline
```
PDF Upload → Text Extraction → Chunking → Embedding → Indexing (Elasticsearch)
                                                              ↓
User Query → Query Rewriting (LLM) → Query Embedding → KNN Search → LLM Response
```

### 1. Chunking
Documents are split into overlapping word-based chunks (`max_words=300`, `overlap=50`). The text is first split into sentences, then words are accumulated until the limit is reached. Overlap ensures context is preserved across chunk boundaries.

### 2. Embedding Generation
Four models were evaluated using cosine similarity benchmarks:
- `all-MiniLM-L6-v2`
- Universal Sentence Encoder (USE)
- Word2Vec
- Mistral-7B-v0.1

**USE** was selected for its strong balance of semantic precision, batch inference speed, and zero-config integration via TensorFlow Hub.

> MiniLM showed marginally higher similarity scores (0.4407 vs 0.4232) but required additional setup. USE remains the default; MiniLM is available as an upgrade path if response quality needs to be maximized.

### 3. Vector Storage (Elasticsearch)
Each chunk is stored as a `{chunk, embedding}` tuple in an Elasticsearch index with the following mapping:
- Field: `embedding` — type `dense_vector`, 512 dims, cosine similarity
- Field: `original_text` — type `text`
- Field: `doc_id` — used to filter chunks per document

### 4. Query Rewriting
Before searching, the user's raw query is passed to LLaMA 3.3 to produce a cleaner, semantically richer query. This improves recall by reducing ambiguity and aligning the query vocabulary with document content.

**Example:**
| Input | Rewritten |
|---|---|
| "Que impacto tiene la influencia humana en los ecosistemas?" | "¿Cuáles son los efectos de la actividad humana en la biodiversidad y el equilibrio de los ecosistemas naturales?" |

### 5. Semantic Search
The rewritten query is embedded using the same USE model and searched against Elasticsearch using **KNN** (`k=5`, `num_candidates=10`). Only the `original_text` field is returned for context assembly.

### 6. Response Generation
The top-5 retrieved chunks are assembled into a prompt together with the original user question. LLaMA 3.3 is instructed to generate a clear, professional, evidence-based answer — and to explicitly state when the provided context is insufficient, reducing hallucinations.

---

## Interface

Built with **Gradio**, the interface supports:
- PDF upload and real-time processing feedback
- Natural language question input
- Formatted answer display
- Friendly error handling at each pipeline step

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | USE | TF Hub integration, batch efficiency, no extra installs |
| LLM | LLaMA 3.3 70B Instruct Turbo | High quality, remote inference, free on Colab |
| Vector DB | Elasticsearch | Reliable KNN, flexible mapping, easy integration |
| Chunking | Word-based + overlap | Simple, effective, avoids mid-sentence cuts |
| Query rewriting | LLM pre-step | Reduces ambiguity, improves retrieval precision |

---

## Challenges

- **Model selection:** Needed a capable LLM that didn't exceed Colab's compute limits. LLaMA 3.3 via remote inference was the solution.
- **Embedding comparison:** Ran cosine similarity benchmarks across 4 models to make an informed choice rather than defaulting to the first option.
- **PDF text quality:** Some documents had encoding issues or complex layouts. Robust extraction and cleaning steps were added using PyMuPDF + spaCy.
- **Dependency conflicts:** Gradio 5+ requires `huggingface_hub>=0.28.1`. Avoid pinning `huggingface_hub` to older versions in `requirements.txt`.

---

## Anti-Hallucination Measures

The final prompt explicitly instructs the LLM to acknowledge when the retrieved context doesn't contain enough information to answer the question, rather than fabricating a response. This was validated by querying the system on topics completely unrelated to the loaded document (e.g., "things to do in New York") — the model correctly flagged the lack of relevant context.

