import streamlit as st
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel 
import chromadb
from chromadb.config import Settings

# -----------------------------
# 0) API AYARLARI
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()
    
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = GenerativeModel("gemini-1.5-flash")

# -----------------------------
# 1) CHROMA DB LOAD (Direkt ChromaDB Client)
# -----------------------------
DB_PATH = "chroma_db"
if not os.path.exists(DB_PATH):
    st.error("❌ HATA: Chroma DB ('chroma_db' klasörü) bulunamadı.")
    st.stop()

@st.cache_resource
def get_chroma_client():
    """ChromaDB'yi direkt yükle - embedding modeli YOK"""
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Koleksiyonu al (varsayılan isim: langchain)
    try:
        collection = client.get_collection(name="langchain")
        return collection
    except Exception as e:
        st.error(f"Koleksiyon bulunamadı: {e}")
        # Tüm koleksiyonları listele
        collections = client.list_collections()
        if collections:
            st.info(f"Mevcut koleksiyonlar: {[c.name for c in collections]}")
            return collections[0]  # İlkini al
        return None

# -----------------------------
# 2) RAG PIPELINE
# -----------------------------
def ask_rag(question):
    """Kullanıcı sorusuna RAG ile cevap verir."""
    
    collection = get_chroma_client()
    if not collection:
        return "❌ Vektör veritabanı yüklenemedi.", []
    
    try:
        # ChromaDB query (embedding yapmadan text araması)
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        
        # Sonuçları işle
        if not results['documents'] or not results['documents'][0]:
            return "❌ İlgili döküman bulunamadı.", []
        
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
        
        # Context oluştur
        context = "\n\n".join(docs)
        
        # Prompt oluştur
        prompt = f"""Sen bir astroloji uzmanısın. Aşağıdaki bilgileri kullanarak soruyu yanıtla.

BAĞLAM:
{context}

SORU: {question}

YANIT (Türkçe ve detaylı):"""
        
        # Gemini API çağrısı
        response = llm.generate_content(prompt)
        
        # Sonuçları formatla
        formatted_results = []
        for doc, meta in zip(docs, metadatas):
            formatted_results.append({
                'content': doc,
                'metadata': meta
            })
        
        return response.text, formatted_results
    
    except Exception as e:
        st.error(f"Arama hatası: {type(e).__name__}")
        st.error(f"Detay: {str(e)}")
        return None, []

# -----------------------------
# 3) STREAMLIT UI
# -----------------------------
st.title("🔮 Astrology RAG Chatbot")
st.write("Astroloji hakkında her şeyi sorabilirsiniz. Gemini + ChromaDB ile güçlendirilmiştir.")

# Debug info
with st.expander("🔧 Sistem Bilgisi"):
    col = get_chroma_client()
    if col:
        st.success(f"✅ Koleksiyon: {col.name}")
        st.info(f"📊 Toplam döküman: {col.count()}")

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
                
                if chunks:
                    with st.expander("🔍 Kaynak Dökümanlar"):
                        for i, c in enumerate(chunks, 1):
                            st.markdown(f"**Kaynak {i}:**")
                            st.text(c['content'][:300] + "...")
                            if c['metadata']:
                                st.caption(f"Metadata: {c['metadata']}")
                            st.divider()
