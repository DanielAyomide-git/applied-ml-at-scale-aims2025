import csv
import pickle

DF_PATH = "data/fitdata_small.pkl"   # <-- update if path differs
LINKS_PATH = "data/ml-32m/ml-32m/links.csv"
OUT_PATH = "data/imdb_to_movielens.pkl"

# -----------------------
# LOAD DF PICKLE
# -----------------------
with open(DF_PATH, "rb") as f:
    df = pickle.load(f)

# handle both object and dict
if isinstance(df, dict):
    if "movie_id_to_idx" in df:
        valid_movies = set(df["movie_id_to_idx"].keys())
    else:
        raise ValueError("fitdata pickle is dict but missing movie_id_to_idx")
else:
    valid_movies = set(df.movie_id_to_idx.keys())

print("Movies in training data:", len(valid_movies))

# -----------------------
# BUILD MAPPING
# -----------------------
imdb_to_ml = {}

with open(LINKS_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movie_id = int(row["movieId"])
        if movie_id not in valid_movies:
            continue

        imdb_raw = row["imdbId"]
        if imdb_raw:
            imdb_id = "tt" + imdb_raw.zfill(7)
            imdb_to_ml[imdb_id] = movie_id

print("IMDb mappings kept:", len(imdb_to_ml))

# -----------------------
# SAVE
# -----------------------
with open(OUT_PATH, "wb") as f:
    pickle.dump(imdb_to_ml, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved:", OUT_PATH)
