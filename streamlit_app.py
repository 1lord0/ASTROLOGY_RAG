import streamlit as st
import os
from google.generativeai import GenerativeModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 1) CHROMA DB LOAD
# -----------------------------

DB_PATH = "chroma_db"

if not os.path.exists(DB_PATH):
    st.error("Chroma DB not found. Make sure 'chroma_db' folder is in the repo.")
else:
    # FakeEmbeddings yerine gerçek BGE-base kullanılacak
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=emb
    )

# -----------------------------
# 2) GEMINI MODEL SETUP
# -----------------------------

import google.generativeai as genai
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

llm = GenerativeModel("gemini-pro")

# -----------------------------
# 3) RAG PIPELINE
# -----------------------------

def ask_rag(question):
    # ❗ Soru embedding artık BGE-base ile yapılacak (Gemini değil)
    q_emb = emb.embed_query(question)

    # Chroma araması
    results = db.similarity_search_by_vector(q_emb, k=3)

    context = "\n\n".join(
        f"---Chunk---\n{doc.page_content}" for doc in results
    )

    prompt = f"""
    Act like you have been a professional astrologer for decades.
    Use ONLY the context below when generating the answer.
    After giving the English answer, translate it into Turkish
    in the SAME tone and style.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    answer = llm.generate_content(prompt)
    return answer.text, results

# -----------------------------
# 4) STREAMLIT UI
# -----------------------------

st.title("🔮 Astrology RAG Chatbot")
st.write("Ask anything about astrology. Powered by Gemini + ChromaDB.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Consulting the stars..."):
        answer, chunks = ask_rag(question)

    st.subheader("🌟 Answer")
    st.write(answer)

    st.subheader("🔍 Retrieved Chunks")
    for i, c in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(c.page_content)
