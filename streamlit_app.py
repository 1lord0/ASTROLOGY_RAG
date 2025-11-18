import streamlit as st
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 0) API VE KÜTÜPHANE AYARLARI
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()
    
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = GenerativeModel("gemini-1.5-flash")

# -----------------------------
# 1) CHROMA DB LOAD
# -----------------------------
DB_PATH = "chroma_db"
if not os.path.exists(DB_PATH):
    st.error("❌ HATA: Chroma DB ('chroma_db' klasörü) bulunamadı.")
    st.stop()

# 🔥 ÖNEMLİ: model_kwargs ile device ayarı
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},  # CPU kullanımını zorla
    encode_kwargs={'normalize_embeddings': True}
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=emb
)

# -----------------------------
# 2) RAG PIPELINE
# -----------------------------
def ask_rag(question):
    """Kullanıcı sorusuna RAG ile cevap verir."""
    
    # Soru embedding'i oluştur
    q_emb_list = emb.embed_documents([question])
    q_emb = q_emb_list[0]
    
    # Benzer dökümanları bul
    results = db.similarity_search_by_vector(q_emb, k=3)
    
    # Context oluştur
    context = "\n\n".join([chunk.page_content for chunk in results])
    
    # Prompt oluştur
    prompt = f"""Sen bir astroloji uzmanısın. Aşağıdaki bilgileri kullanarak soruyu yanıtla.

BAĞLAM:
{context}

SORU: {question}

YANIT (Türkçe ve detaylı):"""
    
    # Gemini API çağrısı
    response = llm.generate_content(prompt)
    
    return response.text, results

# -----------------------------
# 3) STREAMLIT UI
# -----------------------------
st.title("🔮 Astrology RAG Chatbot")
st.write("Astroloji hakkında her şeyi sorabilirsiniz.")

question = st.text_input("Sorunuz:")

if st.button("Sorgula") or question:
    if not question or not question.strip():
        st.warning("⚠️ Lütfen geçerli bir soru girin.")
    else:
        with st.spinner("Yıldızlara danışılıyor..."):
            try:
                answer, chunks = ask_rag(question)
                
                st.subheader("🌟 Cevap")
                st.write(answer)
                
                with st.expander("🔍 Kaynak Dökümanlar"):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Kaynak {i}:**")
                        st.text(c.page_content[:300] + "...")
                        st.divider()
                        
            except Exception as e:
                st.error(f"❌ Bir hata oluştu: {type(e).__name__}")
                st.error(f"Detay: {str(e)}")
