from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load a pretrained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Two sets of text (answers from different times)
texts = [
    "The person appears to be raising their hand",
    "They might be signaling for attention"
]

# 3) Generate embedding vectors for each text
embeddings = model.encode(texts)

# 4) Compute pairwise similarity
sim_matrix = cosine_similarity(embeddings)

print(sim_matrix)