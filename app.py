import os
import re
import pickle
import torch
import faiss
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

with open("faq_data.pkl", "rb") as f:
    faq_data = pickle.load(f)
    faq_df = faq_data["faq_df"]
    vectorizer = faq_data["vectorizer"]
    faq_vectors = faq_data["faq_vectors"]

with open("rag_data.pkl", "rb") as f:
    rag_data = pickle.load(f)
    rag_texts = rag_data["texts"]
    rag_embeddings = rag_data["embeddings"]

index = faiss.IndexFlatL2(rag_embeddings.shape[1])
index.add(rag_embeddings.astype(np.float32))

def get_faq_answer(query):
    query_vec = vectorizer.transform([query])
    sim = cosine_similarity(query_vec, faq_vectors).flatten()
    idx = np.argmax(sim)
    if sim[idx] > 0.7:
        return faq_df.iloc[idx]['answer'], "L1"
    return None, None

def get_rag_answer(query):
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb).astype(np.float32), k=1)
    if D[0][0] < 1.1:
        return rag_texts[I[0][0]], "L2"
    return None, None

def get_llm_answer(query):
    prompt = f"You are a technical support agent. Provide a solution for: {query}. Classify as L3."
    response = llm.generate_content(prompt)
    return response.text, "L3"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json if request.is_json else request.form
    query = data.get("query")

    ans, level = get_faq_answer(query)
    
    if not ans:
        ans, level = get_rag_answer(query)
        
    if not ans:
        ans, level = get_llm_answer(query)

    return jsonify({"answer": ans, "level": level})

if __name__ == "__main__":
    app.run(debug=True)