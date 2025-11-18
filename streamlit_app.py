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
# 1) CHROMA DB LOAD (Embedding Lazy Loading)
# -----------------------------
DB_PATH = "chroma_db"
if not os.path.exists(DB_PATH):
    st.error("❌ HATA: Chroma DB ('chroma_db' klasörü) bulunamadı.")
    st.stop()

# 🔥 Embedding modelini sadece gerektiğinde yükle
@st.cache_resource
def get_embeddings():
    """Embeddings'i cache'le - bir kez yükle"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_vectordb():
    """Vektör veritabanını cache'le"""
    emb = get_embeddings()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=emb
    )

# -----------------------------
# 2) RAG PIPELINE
# -----------------------------
def ask_rag(question):
    """Kullanıcı sorusuna RAG ile cevap verir."""
    
    try:
        db = get_vectordb()
        
        # Direkt text ile arama (embedding hesaplanmış)
        results = db.similarity_search(question, k=3)
        
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
    
    except Exception as e:
        st.error(f"Model yükleme hatası: {str(e)}")
        st.info("💡 Lütfen Python 3.11 kullanın veya packages.txt ekleyin")
        return None, []

# -----------------------------
# 3) STREAMLIT UI
# -----------------------------
st.title("🔮 Astrology RAG Chatbot")
st.write("Astroloji hakkında her şeyi sorabilirsiniz. Gemini + ChromaDB ile güçlendirilmiştir.")

question = st.text_input("Sorunuz:")

if st.button("Sorgula") or question:
    if not question or not question.strip():
        st.warning("⚠️ Lütfen geçerli bir soru girin.")
    else:
        with st.spinner("Yıldızlara danışılıyor..."):
            answer, chunks = ask_rag(question)
            
            if answer:
                st.subheader("🌟 Cevap")
                st.write(answer)
                
                with st.expander("🔍 Kaynak Dökümanlar"):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Kaynak {i}:**")
                        st.text(c.page_content[:300] + "...")
                        st.divider()
