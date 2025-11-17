import streamlit as st
import os
import zipfile
from google.generativeai import GenerativeModel, embed_content
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# -----------------------------
# 1) CHROMA DB LOAD (zip'ten çıkarma yok çünkü sen direkt repo'ya yükledin)
# -----------------------------

DB_PATH = "chroma_db"

if not os.path.exists(DB_PATH):
    st.error("Chroma DB not found. Make sure 'chroma_db' folder is in the repo.")
else:
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=FakeEmbeddings(size=768)
    )

# -----------------------------
# 2) GEMINI MODEL SETUP
# -----------------------------

import google.generativeai as genai
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

llm = GenerativeModel("gemini-pro")

def embed_query(text):
    res = embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return res["embedding"]

# -----------------------------
# 3) RAG PIPELINE
# -----------------------------

def ask_rag(question):
    q_emb = embed_query(question)
    
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
