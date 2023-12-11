# 12/11/2023
# CSC461 – Assignment4 – NLP
# Arooba Masood
# FA20-BSE-092
# The task is related to calculate BoW, TF, TF.IDF vectors and calculation of their similiratities using cosing, euclidean and manhattan differences.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances

sentences = [
    "data science is one of the most important courses in computer science",
    "this is one of the best data science courses",
    "the data scientists perform data analysis"
]

bow_vectorizer = CountVectorizer()
bow_X = bow_vectorizer.fit_transform(sentences)
bow_vectors = bow_X.toarray()

tf_vectorizer = CountVectorizer()
tf_X = tf_vectorizer.fit_transform(sentences)
tf_vectors = tf_X.toarray()

print("Vocabulary:")
print(tf_vectorizer.get_feature_names_out())

print("\nBoW Vectors:")
for i, sentence in enumerate(sentences):
    print(f"S{i + 1}: {bow_vectors[i]}")

print("\nTF Vectors:")
for i, sentence in enumerate(sentences):
    print(f"S{i + 1}: {tf_vectors[i]}")

bow_cosine_sim = cosine_similarity(bow_vectors)
tf_cosine_sim = cosine_similarity(tf_vectors)

print("\nCosine Similarity of BoW Vectors:")
print(bow_cosine_sim)

print("\nCosine Similarity of TF Vectors:")
print(tf_cosine_sim)

bow_manhattan_dist = manhattan_distances(bow_vectors)
tf_manhattan_dist = manhattan_distances(tf_vectors)

print("\nManhattan Distance of BoW Vectors:")
print(bow_manhattan_dist)

print("\nManhattan Distance of TF Vectors:")
print(tf_manhattan_dist)

bow_euclidean_dist = euclidean_distances(bow_vectors)
tf_euclidean_dist = euclidean_distances(tf_vectors)

print("\nEuclidean Distance of BoW Vectors:")
print(bow_euclidean_dist)

print("\nEuclidean Distance of TF Vectors:")
print(tf_euclidean_dist)
print("\n")

tokenized_sentences = [sentence.split() for sentence in sentences]

tf_values = []
for sentence in tokenized_sentences:
    term_frequency = {}
    sentence_length = len(sentence)
    for word in sentence:
        if word not in term_frequency:
            term_frequency[word] = 0
        term_frequency[word] += 1 / sentence_length  
    tf_values.append(term_frequency)

for i, tf in enumerate(tf_values, start=1):
    print(f"TF values for S{i}:")
    for term, value in tf.items():
        print(f"{term}: {value}")
    print()

tf_values = [
    {'data': 0.16666666666666666, 'science': 0.16666666666666666, 'is': 0.16666666666666666, 'one': 0.16666666666666666, 'of': 0.16666666666666666, 'the': 0.16666666666666666, 'most': 0.0, 'important': 0.0, 'courses': 0.0, 'in': 0.0, 'computer': 0.0},
    {'this': 0.16666666666666666, 'is': 0.16666666666666666, 'one': 0.16666666666666666, 'of': 0.16666666666666666, 'the': 0.0, 'best': 0.0, 'data': 0.16666666666666666, 'science': 0.16666666666666666, 'courses': 0.0},
    {'the': 0.2, 'data': 0.2, 'scientists': 0.2, 'perform': 0.2, 'analysis': 0.2}
]

tf_vectors = [[tf.get(term, 0) for term in tf_vectorizer.get_feature_names_out()] for tf in tf_values]

tf_cosine_sim_manual = cosine_similarity(tf_vectors)

print("\nCosine Similarity of Manual TF Vectors:")
print(tf_cosine_sim_manual)

tf_manhattan_dist_manual = manhattan_distances(tf_vectors)
print("\nManhattan Distance of Manual TF Vectors:")

print(tf_manhattan_dist_manual)
tf_euclidean_dist_manual = euclidean_distances(tf_vectors)

print("\nEuclidean Distance of Manual TF Vectors:")
print(tf_euclidean_dist_manual)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
terms = vectorizer.get_feature_names_out()
doc_frequency = np.sum(X.toarray() > 0, axis=0)

total_docs = len(sentences)

idf_values = {term: np.log(total_docs / freq) for term, freq in zip(terms, doc_frequency)}

print("\n\nIDF Values:")
for term, idf in idf_values.items():
    print(f"{term}: {idf}")


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
terms = vectorizer.get_feature_names_out()
doc_frequency = np.sum(X.toarray() > 0, axis=0)

total_docs = len(sentences)

idf_values = {term: np.log(total_docs / freq) for term, freq in zip(terms, doc_frequency)}

tfidf_values = []

for tf in tf_values:
    tfidf = {term: tf[term] * idf_values[term] for term in tf}
    tfidf_values.append(tfidf)

print("\nTF-IDF Values:")
for i, tfidf in enumerate(tfidf_values, start=1):
    print(f"For Sentence {i}: {tfidf}")

cosine_sim = cosine_similarity(tfidf_vectors)
print("\nCosine Similarity:")
print(cosine_sim)

manhattan_dist = manhattan_distances(tfidf_vectors)
print("\nManhattan Distance:")
print(manhattan_dist)

euclidean_dist = euclidean_distances(tfidf_vectors)
print("\nEuclidean Distance:")
print(euclidean_dist)
