from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"

def test_retrieval():
    print("🧠 Loading model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Loading DB...")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    print("✔ Ready.")

    while True:
        query = input("\n🔍 Soru (exit yaz ve çık): ")
        if query.lower() == "exit":
            break

        results = db.similarity_search_with_score(query, k=3)

        print("\n📄 En yakın parçalar ve skorlar:")
        for i, (doc, score) in enumerate(results, 1):

            # distance → similarity
            similarity = 1 / (1 + score)

            print(f"\n--- Result {i} ---")
            print("Distance:", score)
            print("Similarity:", round(similarity, 4))
            print("\nContent:")
            print(doc.page_content[:400])
            print("\n----------------------")

# ❗ EN ÖNEMLİ KISIM — senin eksik olan bölümün
if __name__ == "__main__":
    test_retrieval()
