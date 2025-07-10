"""# Interfaz con Gradio"""

import gradio as gr
import fitz
import os
import spacy
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
from together import Together
from dotenv import load_dotenv
import subprocess
import sys

# Intenta cargar el modelo, y si no está, lo descarga
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Credenciales (desde entorno seguro)
load_dotenv()  # Esto carga las variables del archivo .env

CLOUD_ID = os.getenv("CLOUD_ID")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")



# Conexión a Elasticsearch
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    http_compress=True,
    basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
)

# Cliente de Together AI
client = Together(api_key=TOGETHER_API_KEY)

# Nombre del índice
index_name = "document_chunks"

# Funciones auxiliares
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

def obtener_doc_ids():
    try:
        response = es.search(
            index=index_name,
            size=0,
            aggs={
                "doc_ids": {
                    "terms": {
                        "field": "doc_id.keyword",
                        "size": 100
                    }
                }
            }
        )
        return [bucket["key"] for bucket in response["aggregations"]["doc_ids"]["buckets"]]
    except Exception as e:
        print("Error obteniendo doc_ids:", e)
        return []

def actualizar_dropdown():
    doc_ids = obtener_doc_ids()
    return gr.update(choices=doc_ids, value=None)

def procesar_pdf_gradio(pdf_file):
    print(f"procesar_pdf_gradio recibió: {pdf_file}")
    # resto de la función
    # try:
    #     with open(pdf_file, "rb") as f:
    #         file_bytes = f.read()
    #     text = extract_text_from_pdf(file_bytes)
    # except Exception as e:
    #     return f"Error al procesar el PDF: {e}"

    # chunks = chunk_text(text)
    # vectors = get_embeddings(chunks)

    # if not es.indices.exists(index=index_name):
    #     es.indices.create(
    #         index=index_name,
    #         body={
    #             "mappings": {
    #                 "properties": {
    #                     "doc_id": {"type": "keyword"},
    #                     "original_text": {"type": "text"},
    #                     "embedding": {
    #                         "type": "dense_vector",
    #                         "dims": 512,
    #                         "index": True,
    #                         "similarity": "cosine"
    #                     }
    #                 }
    #             }
    #         }
    #     )

    # doc_id_actual = os.path.basename(pdf_file)
    # for chunk, vector in zip(chunks, vectors):
    #     doc = {
    #         "doc_id": doc_id_actual,
    #         "original_text": chunk,
    #         "embedding": vector.tolist()
    #     }
    #     es.index(index=index_name, document=doc)

    # return f"PDF procesado exitosamente. {len(chunks)} fragmentos indexados."

def responder_pregunta(user_input, selected_doc_id):
    query_vector = get_embedding(user_input)
    print("Vector de la consulta:", selected_doc_id)

    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 10,
            "filter": [
                {"term": {"doc_id.keyword": selected_doc_id}}
            ]
        },
        _source=["original_text", "doc_id"]
    )


    top_chunks = [hit["_source"]["original_text"] for hit in response["hits"]["hits"]]
    context = "\n\n".join(top_chunks)

    final_prompt = f"""
Usá el siguiente contexto extraído de un documento para responder la pregunta del usuario de forma clara, completa y profesional. Si no hay suficiente información en el contexto, decilo explícitamente.

### CONTEXTO DEL DOCUMENTO:
{context}

### PREGUNTA:
{user_input}

### RESPUESTA:
"""

    response_final = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )

    return response_final.choices[0].message.content

# Interfaz de usuario Gradio
with gr.Blocks() as demo:
    gr.Markdown("### 🧠 Subí un PDF y preguntá sobre su contenido")

    with gr.Row():
        upload = gr.File(label="📄 Subí un PDF", file_types=[".pdf"], type="binary")
        upload_btn = gr.Button("Procesar PDF")

    status = gr.Textbox(label="Estado", interactive=False)

    with gr.Row():
        doc_id_dropdown = gr.Dropdown(label="Seleccioná un documento", choices=[])
        refresh_btn = gr.Button("🔄 Recargar lista")

    user_input = gr.Textbox(label="Pregunta")
    answer = gr.Textbox(label="Respuesta generada", lines=10)
    ask_btn = gr.Button("Preguntar")

    # Vinculaciones de eventos
    refresh_btn.click(fn=actualizar_dropdown, inputs=[], outputs=[doc_id_dropdown])

    upload_btn.click(fn=procesar_pdf_gradio, inputs=[upload], outputs=[status])

    # ask_btn.click(fn=responder_pregunta, inputs=[user_input, doc_id_dropdown], outputs=[answer])

demo.launch(debug=True, share=True)