import csv


class FitData:
    def __init__(self, ratings_csv, movies_csv=None):
        self.ratings_csv = ratings_csv
        self.movies_csv = movies_csv

        # Mappings
        self.user_id_to_idx = {}
        self.idx_to_user_id = []
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = []

        self.movie_id_to_title = {}  # store movie titles

        # Ratings data
        self.data_by_user = []
        self.data_by_movie = []

    def load(self):
        user_set = set()
        movie_set = set()

        # First pass: collect all unique IDs from ratings
        with open(self.ratings_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                user_set.add(int(row[0]))
                movie_set.add(int(row[1]))

        # Create sorted lists of IDs
        self.idx_to_user_id = sorted(list(user_set))
        self.idx_to_movie_id = sorted(list(movie_set))

        # Create mappings
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.idx_to_user_id)}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.idx_to_movie_id)}

        # Initialize rating containers
        self.data_by_user = [[] for _ in range(len(self.idx_to_user_id))]
        self.data_by_movie = [[] for _ in range(len(self.idx_to_movie_id))]

        # Second pass: fill ratings
        with open(self.ratings_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                user_id = int(row[0])
                movie_id = int(row[1])
                rating = float(row[2])

                u_idx = self.user_id_to_idx[user_id]
                m_idx = self.movie_id_to_idx[movie_id]

                self.data_by_user[u_idx].append((movie_id, rating))
                self.data_by_movie[m_idx].append((user_id, rating))

        # Load movie titles if movies CSV is provided
        if self.movies_csv:
            with open(self.movies_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    movie_id = int(row[0])
                    title = row[1]
                    self.movie_id_to_title[movie_id] = title

    def get_user_ratings(self, user_id):
        u_idx = self.user_id_to_idx[user_id]
        return self.data_by_user[u_idx]

    def get_movie_ratings(self, movie_id):
        m_idx = self.movie_id_to_idx[movie_id]
        return self.data_by_movie[m_idx]

    def num_users(self):
        return len(self.idx_to_user_id)

    def num_movies(self):
        return len(self.idx_to_movie_id)

    def movie_titles(self, movie_ids):
        """Return titles for a list of movie IDs, fallback to ID string if missing."""
        return [self.movie_id_to_title.get(mid, str(mid)) for mid in movie_ids]
