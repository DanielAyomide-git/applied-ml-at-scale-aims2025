import pickle
import numpy as np
from pathlib import Path

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = Path("./app/models/als_model_k10.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# METADATA & PRECOMPUTE
# -----------------------------
idx_to_imdb_id = model.idx_to_imdb_id
imdb_to_idx = {imdb: i for i, imdb in enumerate(idx_to_imdb_id) if imdb is not None}

# IMPORTANT: We must pre-calculate norms for TRUE Cosine Similarity
item_norms = np.linalg.norm(model.V, axis=1)
item_norms[item_norms == 0] = 1e-9 # Prevent division by zero

# Popularity Filter: 
# If your model doesn't have movie_counts, we assume 101 to pass the filter, 
# but for better results, you should pass actual counts here.
movie_counts = getattr(model, "movie_counts", np.ones(model.V.shape[0]) * 150)
MIN_RATINGS = 100

# -----------------------------
# COSINE SIMILARITY RECOMMENDATION
# -----------------------------
# recommender.py

def recommend_for_user_profile(user_ratings_dict, top_n=50):
    rated_indices = []
    user_vec = np.zeros(model.V.shape[1])

    for imdb_id, rating in user_ratings_dict.items():
        idx = imdb_to_idx.get(imdb_id)
        if idx is None: continue
        rated_indices.append(idx)
        weight = float(rating) - 3.0
        user_vec += model.V[idx] * weight

    if len(rated_indices) < 1:
        popular_ids = get_top_popular_movies(top_n)
        return [{"imdb_id": imdb_id} for imdb_id in popular_ids]

    user_norm = np.linalg.norm(user_vec)
    if user_norm == 0: user_norm = 1e-9
    
    dot_products = model.V @ user_vec
    similarities = dot_products / (item_norms * user_norm)

    # --- THE FIX: NOISE FILTERING ---
    # 1. Filter out movies the user already rated
    for idx in rated_indices:
        similarities[idx] = -2.0

    # 2. Filter out "Short Vectors" (Proxy for unpopular movies)
    # Most well-rated movies in ALS have a norm between 0.1 and 2.0
    # Movies like "The Droving" often have extreme norms because they are outliers.
    similarities[item_norms < 0.1] = -2.0 

    ranked_indices = np.argsort(similarities)[::-1]
    
    # We take a much larger slice (100) and let main.py filter the genres
    rec_indices = ranked_indices[:100] 

    recs = []
    for i in rec_indices:
        if similarities[i] < -1: break
        imdb_id = idx_to_imdb_id[i]
        if imdb_id:
            recs.append({"imdb_id": imdb_id})

    return recs
def get_top_popular_movies(top_n=20):
    # Sort by movie_counts if available
    top_indices = np.argsort(movie_counts)[-top_n:][::-1]
    return [
        idx_to_imdb_id[idx]
        for idx in top_indices
        if idx_to_imdb_id[idx] is not None
    ]