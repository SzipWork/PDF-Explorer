# Imports
import os
import gradio as gr
import PyPDF2
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# PDF Extraction
def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)


# Split long text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


# Embed text chunks
def embed_texts(texts):
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        embeddings.append(np.array(response['embedding']))
    return np.vstack(embeddings).astype("float32")


# Build FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# Retrieve relevant chunks from FAISS
def retrieve_similar_chunks(question, faiss_index, chunks, top_k=3):
    response = genai.embed_content(
        model="models/embedding-001",
        content=question
    )
    query_vec = np.array(response['embedding']).reshape(1, -1).astype("float32")

    D, I = faiss_index.search(query_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results


# Generate final answer using Gemini
def generate_answer(question, context):
    prompt = f"Use the context below to answer the question.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()


# Preprocess PDFs 
pdf1_text = extract_text_from_pdf("AIAYU.pdf")
pdf2_text = extract_text_from_pdf("mcs.pdf")

chunks1 = split_text(pdf1_text)
chunks2 = split_text(pdf2_text)
all_chunks = chunks1 + chunks2

embeddings = embed_texts(all_chunks)
faiss_index = create_faiss_index(embeddings)


# Gradio Chatbot Function
def respond(message, history: list[dict[str, str]], system_message, max_tokens, temperature, top_p):
    # Retrieve context
    relevant_chunks = retrieve_similar_chunks(message, faiss_index, all_chunks)
    context = "\n".join(relevant_chunks)

    # Response(answer)
    answer = generate_answer(message, context)
    return answer


# Gradio Chatbot UI 
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant that answers using PDF knowledge.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)


with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown("## 📄 PDF QA Chatbot")  
        gr.Markdown("Ask questions based on your PDFs.")  
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
