from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from clean import clean_text
import os

PDF_PATH = "data/astrology.pdf"
DB_DIR = "chroma_db"

def build_index():

    print("Loading PDF...")
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()

    for d in docs:
        d.page_content = clean_text(d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print("✅ DONE. Saved to chroma_db/")


if __name__ == "__main__":
    build_index()
