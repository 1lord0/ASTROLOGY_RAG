import streamlit as st
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel
import json

# -----------------------------
# 0) API AYARLARI
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()
    
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = GenerativeModel("gemini-1.5-flash")

# -----------------------------
# 1) BASIT JSON VEKTÖR DEPOSU
# -----------------------------
@st.cache_data
def load_documents():
    """Chroma DB yerine basit JSON kullan"""
    json_path = "documents.json"
    
    # Eğer JSON yoksa, chroma_db'den oku (bir kerelik)
    if not os.path.exists(json_path):
        st.warning("⚠️ documents.json bulunamadı. Lütfen lokal olarak oluşturun.")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -----------------------------
# 2) BASIT ARAMA FONKSİYONU
# -----------------------------
def simple_search(query, documents, k=3):
    """Keyword-based basit arama"""
    query_words = set(query.lower().split())
    
    scores = []
    for doc in documents:
        doc_words = set(doc['content'].lower().split())
        score = len(query_words & doc_words)  # Ortak kelime sayısı
        scores.append((score, doc))
    
    # Skorlara göre sırala
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scores[:k]]

# -----------------------------
# 3) RAG PIPELINE
# -----------------------------
def ask_rag(question):
    """Kullanıcı sorusuna RAG ile cevap verir."""
    
    docs = load_documents()
    if not docs:
        return "❌ Dökümanlar yüklenemedi. Lütfen documents.json dosyasını oluşturun.", []
    
    # Basit arama
    results = simple_search(question, docs, k=3)
    
    if not results:
        return "❌ İlgili döküman bulunamadı.", []
    
    # Context oluştur
    context = "\n\n".join([doc['content'] for doc in results])
    
    # Prompt oluştur
    prompt = f"""Sen bir astroloji uzmanısın. Aşağıdaki bilgileri kullanarak soruyu yanıtla.

BAĞLAM:
{context}

SORU: {question}

YANIT (Türkçe ve detaylı):"""
    
    try:
        # Gemini API çağrısı
        response = llm.generate_content(prompt)
        return response.text, results
    except Exception as e:
        st.error(f"API Hatası: {str(e)}")
        return None, []

# -----------------------------
# 4) STREAMLIT UI
# -----------------------------
st.title("🔮 Astrology RAG Chatbot")
st.write("Astroloji hakkında her şeyi sorabilirsiniz.")

# Debug info
with st.expander("🔧 Sistem Bilgisi"):
    docs = load_documents()
    st.info(f"📊 Toplam döküman: {len(docs)}")
    if docs:
        st.success("✅ Dökümanlar yüklendi")

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
                            st.divider()
