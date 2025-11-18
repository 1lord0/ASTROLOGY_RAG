import streamlit as st
import json
import os
import google.generativeai as genai
from google.generativeai import GenerativeModel
from deep_translator import GoogleTranslator

# -----------------------------
# 0) API AYARLARI
# -----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
llm = GenerativeModel("gemini-2.5-flash")  # Güncel model

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
# 2) TÜRKÇE → İNGİLİZCE ÇEVİRİ (ÜCRETSİZ)
# -----------------------------

# Astroloji terimleri sözlüğü
ASTROLOGY_TERMS = {
    # Burçlar
    "koç": "aries",
    "boğa": "taurus",
    "ikizler": "gemini",
    "yengeç": "cancer",
    "aslan": "leo",
    "başak": "virgo",
    "terazi": "libra",
    "akrep": "scorpio",
    "yay": "sagittarius",
    "oğlak": "capricorn",
    "kova": "aquarius",
    "balık": "pisces",
    
    # Astroloji terimleri
    "yükselen": "ascendant",
    "ay burcu": "moon sign",
    "güneş burcu": "sun sign",
    "yükselen burcu": "rising sign",
    "astroloji": "astrology",
    "burç": "zodiac sign",
    "burcu": "sign",
    "harita": "chart",
    "natal": "natal",
    "transit": "transit",
    "evler": "houses",
    "gezegenler": "planets",
    "aspects": "aspects",
    "retrograd": "retrograde",
}

@st.cache_data(ttl=3600)  # 1 saat cache
def translate_to_english(turkish_text):
    """Türkçe soruyu İngilizce'ye çevir (astroloji terimleriyle)"""
    try:
        # Önce astroloji terimlerini değiştir
        text_lower = turkish_text.lower()
        translated_terms = turkish_text
        
        for tr_term, en_term in ASTROLOGY_TERMS.items():
            if tr_term in text_lower:
                # Kelime sınırlarını kontrol et (başında/sonunda boşluk veya noktalama)
                import re
                pattern = r'\b' + re.escape(tr_term) + r'\b'
                translated_terms = re.sub(pattern, en_term, translated_terms, flags=re.IGNORECASE)
        
        # Sonra Google Translate ile geri kalanı çevir
        translator = GoogleTranslator(source='tr', target='en')
        english_text = translator.translate(translated_terms)
        
        return english_text
    except Exception as e:
        st.warning(f"⚠️ Çeviri hatası: {str(e)}")
        return turkish_text  # Hata durumunda orijinal metni döndür

# -----------------------------
# 3) ARAMA FONKSİYONU
# -----------------------------
def search_documents(query, documents, k=3):
    """Basit keyword-based arama (İngilizce query ile)"""
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
# 4) RAG FONKSİYONU
# -----------------------------
def ask_rag(question):
    """Soru-cevap sistemi (Türkçe soru → İngilizce arama → Türkçe cevap)"""
    
    # Dökümanları yükle
    docs = load_documents()
    if not docs:
        return "❌ Dökümanlar yüklenemedi.", []
    
    # Türkçe soruyu İngilizce'ye çevir
    with st.spinner("🔄 Soru İngilizce'ye çevriliyor..."):
        english_question = translate_to_english(question)
        st.info(f"🔍 Arama sorgusu: {english_question}")
    
    # İlgili dökümanları bul (İngilizce query ile)
    relevant_docs = search_documents(english_question, docs, k=3)
    
    if not relevant_docs:
        return "❌ Sorunuzla ilgili bilgi bulunamadı. Lütfen farklı kelimeler deneyin.", []
    
    # Context oluştur
    context = "\n\n---\n\n".join([doc['content'] for doc in relevant_docs])
    
    # Prompt (Türkçe cevap isteyeceğiz)
    prompt = f"""Sen bir astroloji uzmanısın. Aşağıdaki İngilizce bilgileri kullanarak soruyu Türkçe olarak yanıtla.

BAĞLAM (İngilizce):
{context}

SORU (Türkçe): {question}

YANIT (detaylı ve Türkçe):"""
    
    try:
        # Gemini'ye sor
        response = llm.generate_content(prompt)
        return response.text, relevant_docs
    
    except Exception as e:
        st.error(f"API Hatası: {str(e)}")
        return None, []

# -----------------------------
# 5) STREAMLIT ARAYÜZÜ
# -----------------------------
st.set_page_config(
    page_title="Astrology RAG Chatbot",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Astrology RAG Chatbot")
st.markdown("Astroloji hakkındaki sorularınız rag ile kitaptan getirilecektir.Cevap gemini ile türetiliyor")

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
    - ✅ Türkçe soru sorun
    - 🔄 Otomatik İngilizce'ye çevrilir
    - 🌟 Türkçe cevap alırsınız
    
    **Örnek sorular:**
    - Koç burcunun özellikleri nelerdir?
    - Aşağan yayın burcu nedir?
    - Yükselen burcun etkisi nedir?
    """)

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input(
        "Sorunuzu Türkçe yazın:",
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
    "Powered by Google Gemini 2.5 Flash 🚀 | Türkçe Çeviri: Google Translate 🇹🇷"
    "</div>",
    unsafe_allow_html=True
)

