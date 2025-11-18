import streamlit as st
import json
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel

# -----------------------------
# 0) API AYARLARI
# --------------- ------------- 
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = GenerativeModel("gemini-2.5-flash")

# -----------------------------
# 1) DÖKÜMAN YÜKLEME
# -----------------------------
@st.cache_data
def load_documents():
    """JSON'dan dökümanları yükle"""
    json_path = "documents.json"
    
    if not os.path.exists(json_path):
        st.error(f"❌ {json_path} bulunamadı! Lütfen build_index.py çalıştırın.")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    return docs

# -----------------------------
# 2) ARAMA FONKSİYONU
# -----------------------------
def search_documents(query, documents, k=3):
    """Basit keyword-based arama"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Her döküman için skor hesapla
    scores = []
    for doc in documents:
        content_lower = doc['content'].lower()
        
        # Skor 1: Tam eşleşme
        exact_match = query_lower in content_lower
        
        # Skor 2: Kelime eşleşmeleri
        doc_words = set(content_lower.split())
        word_matches = len(query_words & doc_words)
        
        # Toplam skor
        score = (100 if exact_match else 0) + word_matches
        
        scores.append((score, doc))
    
    # Skorlara göre sırala
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # En iyi k tanesini döndür
    return [doc for score, doc in scores[:k] if score > 0]

# -----------------------------
# 3) RAG FONKSİYONU
# -----------------------------
def ask_rag(question):
# ---------------------------------
# 1) TÜRKÇE SORUYU İNGİLİZCEYE ÇEVİR (LITERAL)
# ---------------------------------
translate_prompt = f"""
Translate the following Turkish sentence into English EXACTLY word for word.
Do NOT rewrite, shorten, expand, paraphrase, or change the structure.
Do NOT add or remove any meaning.
Return ONLY the literal English translation.

TURKISH:
{question}

ENGLISH (literal):
"""

try:
    translated_question = llm.generate_content(translate_prompt).text.strip()
except:
    translated_question = question  # fallback

    
    # Dökümanları yükle
    docs = load_documents()
    if not docs:
        return "❌ Dökümanlar yüklenemedi.", []
    
    # İlgili dökümanları bul
    relevant_docs = search_documents(question, docs, k=3)
    
    if not relevant_docs:
        return "❌ Sorunuzla ilgili bilgi bulunamadı. Lütfen farklı kelimeler deneyin.", []
    
    # Context oluştur
    context = "\n\n---\n\n".join([doc['content'] for doc in relevant_docs])
    
    # Prompt
    prompt = f"""Sen bir astroloji uzmanısın. Aşağıdaki bilgileri kullanarak soruyu Türkçe olarak yanıtla.
 

BAĞLAM:
{context}

SORU: {question}

YANIT (detaylı ve Türkçe):"""
    
    try:
        # Gemini'ye sor
        response = llm.generate_content(prompt)
        return response.text, relevant_docs
    
    except Exception as e:
        st.error(f"API Hatası: {str(e)}")
        return None, []

# -----------------------------
# 4) STREAMLIT ARAYÜZÜ
# -----------------------------
st.set_page_config(
    page_title="Astrology RAG Chatbot",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Astrology RAG Chatbot")
st.markdown("Astroloji hakkında her şeyi sorun! **Gemini AI** ile güçlendirilmiştir.")

# Sidebar - Sistem bilgileri
with st.sidebar:
    st.header("📊 Sistem Bilgileri")
    docs = load_documents()
    st.metric("Toplam Döküman", len(docs))
    
    if docs:
        st.success("✅ Sistem Hazır")
        total_chars = sum(len(d['content']) for d in docs)
        st.info(f"📝 Toplam Karakter: {total_chars:,}")
    else:
        st.error("❌ Dökümanlar yüklenemedi")
    
    st.markdown("---")
    st.markdown("### 💡 İpuçları")
    st.markdown("""
    - Spesifik sorular sorun
    - Burç isimleri kullanın
    - Astroloji terimleri ekleyin
    """)

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input(
        "Sorunuzu yazın:",
        placeholder="Örn: Koç burcunun özellikleri nelerdir?"
    )

with col2:
    search_button = st.button("🔍 Sorgula", type="primary", use_container_width=True)

# Sorgulama
if search_button or (question and len(question) > 3):
    if not question or not question.strip():
        st.warning("⚠️ Lütfen geçerli bir soru girin.")
    else:
        with st.spinner("🌟 Yıldızlara danışılıyor..."):
            answer, chunks = ask_rag(question)
            
            if answer:
                # Cevap
                st.markdown("## 🌟 Cevap")
                st.markdown(answer)
                
                # Kaynaklar
                if chunks:
                    st.markdown("---")
                    with st.expander("📚 Kaynak Dökümanlar", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"### Kaynak {i}")
                            st.text_area(
                                f"chunk_{i}",
                                chunk['content'],
                                height=150,
                                label_visibility="collapsed"
                            )
                            st.caption(f"Chunk ID: {chunk['id']}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Google Gemini 1.5 Flash 🚀"
    "</div>",
    unsafe_allow_html=True
)




