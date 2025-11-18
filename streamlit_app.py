import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import GenerativeModel # Kalsın ama kullanılmayacak

# -----------------------------
# 0) API VE KÜTÜPHANE AYARLARI
# -----------------------------

# API Anahtarını yükle ve client'ı yapılandır
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY, Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()
    
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client = genai.Client() # Doğrudan API Client'ı oluşturuluyor

# -----------------------------
# 1) CHROMA DB LOAD
# -----------------------------

DB_PATH = "chroma_db"

if not os.path.exists(DB_PATH):
    st.error("Chroma DB not found. Lütfen 'build_index.py' dosyasını çalıştırın.")
    st.stop()

# 🛑 ÖNEMLİ DEĞİŞİKLİK: build_index.py ile AYNI modeli kullanıyoruz
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=emb
)

# -----------------------------
# 2) RAG PIPELINE
# -----------------------------

def ask_rag(question):
    # Soru embedding
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

    # 🛑 ÖNEMLİ DEĞİŞİKLİK: Doğrudan genai.Client() üzerinden çağırma
    answer = client.models.generate_content(
        model="gemini-1.5-flash", # Hızlı ve stabil bir model
        contents=prompt
    )
    
    return answer.text, results

# -----------------------------
# 3) STREAMLIT UI
# -----------------------------

st.title("🔮 Astrology RAG Chatbot")
st.write("Ask anything about astrology. Powered by Gemini + ChromaDB.")

question = st.text_input("Your question:")

if question:
    # Boş sorgu kontrolü
    if not question.strip():
        st.warning("Please enter a non-empty question.")
    else:
        with st.spinner("Consulting the stars..."):
            try:
                answer, chunks = ask_rag(question)

                st.subheader("🌟 Answer")
                st.write(answer)

                st.subheader("🔍 Retrieved Chunks")
                for i, c in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(c.page_content)
                    
            except Exception as e:
                # API hatasını kullanıcı dostu bir şekilde göster
                st.error(f"An error occurred while consulting Gemini. Check your API key and connection.")
                # st.exception(e) # Streamlit Cloud'da detayları göstermek riskli olabilir
