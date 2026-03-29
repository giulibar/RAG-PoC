import gradio as gr
import fitz
import os
import spacy
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
from groq import Groq
import subprocess
import sys

# Load spaCy model, download if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

    
# Load USE model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load credentials from HF Secrets (Settings → Variables and Secrets in your Space)
CLOUD_ID = os.getenv("CLOUD_ID")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Elasticsearch connection
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    http_compress=True,
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
)

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Index name
index_name = "document_chunks"


# ── Helper functions ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def chunk_text(text, max_words=300, overlap=50):
    sentences = split_into_sentences(text)
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_length = len(current_chunk)
        current_chunk.extend(words)
        current_length += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embeddings(text_chunks):
    return use_model(text_chunks).numpy()

def get_embedding(text):
    return use_model([text])[0].numpy()

def get_doc_ids():
    try:
        response = es.search(
            index=index_name,
            size=0,
            aggs={
                "doc_ids": {
                    "terms": {
                        "field": "doc_id",
                        "size": 100
                    }
                }
            }
        )
        return [bucket["key"] for bucket in response["aggregations"]["doc_ids"]["buckets"]]
    except Exception as e:
        print("Error retrieving doc_ids:", e)
        return []

def update_dropdown():
    doc_ids = get_doc_ids()
    if not doc_ids:
        doc_ids = ["No documents indexed"]
    return gr.update(choices=doc_ids, value=None)

def process_pdf_gradio(pdf_file_path):
    try:
        if not pdf_file_path:
            return "No file uploaded."

        with open(pdf_file_path, "rb") as f:
            file_bytes = f.read()

        text = extract_text_from_pdf(file_bytes)

    except Exception as e:
        return f"Error processing PDF: {e}"

    chunks = chunk_text(text)
    vectors = get_embeddings(chunks)

    # Create index if it doesn't exist
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "original_text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 512,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        )

    doc_id_actual = os.path.splitext(os.path.basename(pdf_file_path))[0]

    for chunk, vector in zip(chunks, vectors):
        doc = {
            "doc_id": doc_id_actual,
            "original_text": chunk,
            "embedding": vector.tolist()
        }
        es.index(index=index_name, document=doc)

    return f"PDF processed successfully. {len(chunks)} chunks indexed with ID '{doc_id_actual}'."

def answer_question(user_input, selected_doc_id):
    if not selected_doc_id or selected_doc_id == "No documents indexed":
        return "Please select a document first."
    if not user_input.strip():
        return "Please enter a question."

    query_vector = get_embedding(user_input)

    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 10,
            "filter": [
                {"term": {"doc_id": selected_doc_id}}
            ]
        },
        _source=["original_text", "doc_id"]
    )

    top_chunks = [hit["_source"]["original_text"] for hit in response["hits"]["hits"]]
    context = "\n\n".join(top_chunks)

    final_prompt = f"""
Use the following document context to answer the user's question clearly, thoroughly, and professionally. If the context doesn't provide enough information, say so explicitly.

### DOCUMENT CONTEXT:
{context}

### QUESTION:
{user_input}

### ANSWER:
"""

    response_final = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )

    return response_final.choices[0].message.content


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("### 🧠 Upload a PDF and ask questions about its content")

    with gr.Row():
        upload = gr.File(label="📄 Upload a PDF", file_types=[".pdf"], type="filepath")
        upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        doc_id_dropdown = gr.Dropdown(label="Select a document", choices=[])
        refresh_btn = gr.Button("🔄 Refresh list")

    user_input = gr.Textbox(label="Your question")
    answer = gr.Textbox(label="Generated answer", lines=10)
    ask_btn = gr.Button("Ask")

    refresh_btn.click(fn=update_dropdown, inputs=[], outputs=[doc_id_dropdown])
    upload_btn.click(fn=process_pdf_gradio, inputs=[upload], outputs=[status])
    ask_btn.click(fn=answer_question, inputs=[user_input, doc_id_dropdown], outputs=[answer])

# HF Spaces manages the server — no share=True, no debug=True
demo.launch()