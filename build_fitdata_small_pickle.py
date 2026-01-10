import os, csv, pickle
import numpy as np
from collections import defaultdict

DATA_DIR = "data/ml-32m/ml-32m"
RATINGS = os.path.join(DATA_DIR, "ratings.csv")
MOVIES = os.path.join(DATA_DIR, "movies.csv")

OUT = "data/fitdata_tiny.pkl"

MAX_USERS = 5_000
MAX_MOVIES = 1_000
MIN_RATINGS_PER_MOVIE = 100

print("Counting ratings per movie...")

movie_counts = defaultdict(int)
user_counts = defaultdict(int)

with open(RATINGS, encoding="utf-8") as f:
    r = csv.reader(f)
    next(r)
    for u, m, *_ in r:
        movie_counts[int(m)] += 1
        user_counts[int(u)] += 1

# keep popular movies
top_movies = sorted(
    [m for m in movie_counts if movie_counts[m] >= MIN_RATINGS_PER_MOVIE],
    key=lambda m: movie_counts[m],
    reverse=True
)[:MAX_MOVIES]

# keep active users
top_users = sorted(
    user_counts,
    key=lambda u: user_counts[u],
    reverse=True
)[:MAX_USERS]

movie_set = set(top_movies)
user_set = set(top_users)

print("Movies:", len(movie_set), "Users:", len(user_set))

# id mappings
movie_id_to_idx = {m:i for i,m in enumerate(top_movies)}
user_id_to_idx = {u:i for i,u in enumerate(top_users)}

# containers
data_by_user = [[] for _ in range(len(top_users))]
data_by_movie = [[] for _ in range(len(top_movies))]

print("Collecting ratings...")

with open(RATINGS, encoding="utf-8") as f:
    r = csv.reader(f)
    next(r)
    for u, m, rgt, *_ in r:
        u = int(u); m = int(m); rgt = float(rgt)
        if u in user_set and m in movie_set:
            ui = user_id_to_idx[u]
            mi = movie_id_to_idx[m]
            data_by_user[ui].append((mi, rgt))
            data_by_movie[mi].append((ui, rgt))

# titles
movie_titles = {}

with open(MOVIES, encoding="utf-8") as f:
    r = csv.reader(f)
    next(r)
    for mid, title, *_ in r:
        mid = int(mid)
        if mid in movie_set:
            movie_titles[mid] = title

# convert to numpy (huge memory save)
data_by_user = [np.array(x, dtype=np.float32) for x in data_by_user]
data_by_movie = [np.array(x, dtype=np.float32) for x in data_by_movie]

obj = {
    "movie_id_to_idx": movie_id_to_idx,
    "user_id_to_idx": user_id_to_idx,
    "idx_to_movie_id": top_movies,
    "idx_to_user_id": top_users,
    "data_by_user": data_by_user,
    "data_by_movie": data_by_movie,
    "movie_titles": movie_titles
}

print("Saving pickle...")
with open(OUT, "wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved:", OUT)
