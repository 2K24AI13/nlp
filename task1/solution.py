import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Task 1: Bag of Words Matrix (Customer Reviews) ---
corpus_t1 = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]
vectorizer_t1 = CountVectorizer(stop_words='english')
X_t1 = vectorizer_t1.fit_transform(corpus_t1)
df_t1 = pd.DataFrame(X_t1.toarray(), columns=vectorizer_t1.get_feature_names_out())

print("--- Task 1: BoW DataFrame ---")
print(df_t1)
print("\n" + "="*50 + "\n")

# --- Task 2: Document Search Engine & Relevance Ranking ---
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]
query = ["machine learning algorithms for data"]

vectorizer_t2 = CountVectorizer()
doc_matrix = vectorizer_t2.fit_transform(documents)
query_matrix = vectorizer_t2.transform(query)

similarities = cosine_similarity(query_matrix, doc_matrix).flatten()
ranked_docs = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)

print("--- Task 2: Ranked Documents ---")
for idx, score in ranked_docs:
    print(f"Doc {idx+1} (Score: {score:.4f}): {documents[idx]}")