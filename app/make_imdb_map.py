import csv
import pickle

mapping = {}

with open("data/ml-32m/links.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        imdb = "tt" + row["imdbId"]
        mapping[imdb] = int(row["movieId"])

with open("data/imdb_to_movielens.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("Saved imdb → movielens mapping")
