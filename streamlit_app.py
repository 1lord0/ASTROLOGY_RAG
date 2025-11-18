import streamlit as st
import os
# google-genai kütüphanesi için doğru importlar
import google.generativeai as genai
from google.generativeai import GenerativeModel 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 0) API VE KÜTÜPHANE AYARLARI
# -----------------------------

# API Anahtarını yükle ve yapılandır
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ HATA: 'GEMINI_API_KEY', Streamlit Secrets'ta tanımlanmalıdır.")
    st.stop()
    
# genai'yi API anahtarıyla yapılandır
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🛑 1. Hata Çözümü: GenerativeModel kullanılıyor (AttributeError'ı çözer)
# client = genai.Client() satırı silindi.
# Hızlı ve stabil bir model kullanıyoruz.
llm = GenerativeModel("gemini-1.5-flash")

# -----------------------------
# 1) CHROMA DB LOAD
# -----------------------------

DB_PATH = "chroma_db"

if not os.path.exists(DB_PATH):
    st.error("❌ HATA: Chroma DB ('chroma_db' klasörü) bulunamadı. Lütfen 'build_index.py' dosyasını çalıştırın.")
    st.stop()

# 🛑 2. Hata Çözümü: Veri yükleme kodu ile aynı modeli kullanıyoruz (InvalidArgumentError'ı çözer)
# Lütfen build_index.py dosyanızda da BAAI/bge-base-en-v1.5 kullandığınızdan emin olun.
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

    # 🛑 API Çağrısı: Oluşturulan llm nesnesi kullanılıyor.
    answer = llm.generate_content(prompt)
    
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
        st.warning("Lütfen boş olmayan bir soru girin.")
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
                # API hatalarını daha genel yakalar
                st.error(f"❌ Bir hata oluştu. API anahtarınızın geçerli olduğunu veya kota limitinizi kontrol edin. Detay: {type(e).__name__}")
