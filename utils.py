import csv
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, prange
import math


class FitData:
    def __init__(self, ratings_csv, movies_csv=None):
        self.ratings_csv = ratings_csv
        self.movies_csv = movies_csv

        # Mappings
        self.user_id_to_idx = {}
        self.idx_to_user_id = []
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = []
        self.movie_id_to_genres = {}

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
                    genres = row[2].split("|") if row[2] != "(no genres listed)" else []
                    self.movie_id_to_title[movie_id] = title
                    self.movie_id_to_genres[movie_id] = genres

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


# ---------------------------
# Create figures folder
# ---------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_eda_summary_pdf(
    scores,
    user_rating_counts,
    movie_rating_counts,
    movie_counter,
    movie_ids,
    filename="eda_summary_2x2.pdf",
):
    """
    Creates a single PDF with:
    (1) User activity histogram
    (2) Rating distribution
    (3) Top-20 rated movies
    (4) Power-law (degree distribution for both users and movies)
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --------------------------------------------------
    # (1) User activity histogram (log-x)
    # --------------------------------------------------
    bins = np.logspace(0, math.ceil(math.log10(user_rating_counts.max() + 1)), 40)
    ax1.hist(user_rating_counts, bins=bins, edgecolor="black")
    ax1.set_xscale("log")
    ax1.set_xlabel("Ratings per user (log scale)")
    ax1.set_ylabel("Number of users")
    ax1.set_title("a.")

    # --------------------------------------------------
    # (2) Rating distribution
    # --------------------------------------------------
    rating_bins = np.arange(
        np.floor(scores.min()) - 0.5, np.ceil(scores.max()) + 0.5, 0.5
    )
    ax2.hist(scores, bins=rating_bins, edgecolor="black")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Count")
    ax2.set_title("b.")

    # --------------------------------------------------
    # (3) Top-20 rated movies
    # --------------------------------------------------
    top20 = movie_counter.most_common(20)
    movie_labels = [f"M{movie_ids[i]}" for i, _ in top20]
    movie_counts = [c for _, c in top20]
    y_pos = np.arange(len(movie_labels))

    ax3.barh(y_pos, movie_counts, edgecolor="black")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(movie_labels)
    ax3.invert_yaxis()
    ax3.set_xlabel("Number of ratings")
    ax3.set_title("c.")

    # --------------------------------------------------
    # (4) Power-law / degree distribution for both users & movies
    # --------------------------------------------------
    # Movies
    movie_deg, movie_freq = np.unique(movie_rating_counts, return_counts=True)
    ax4.scatter(movie_deg, movie_freq, s=12, color="green", label="Movies", alpha=0.8)
    # Users
    user_deg, user_freq = np.unique(user_rating_counts, return_counts=True)
    ax4.scatter(user_deg, user_freq, s=12, color="blue", label="Users", alpha=0.8)

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Degree (number of ratings)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("d.")
    ax4.legend()

    # --------------------------------------------------
    # Layout & save
    # --------------------------------------------------
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_path}")


# --------------------------
# NUMBA OPTIMIZED UPDATES
# --------------------------


# --------------------------
# Numba-optimized update functions
# --------------------------


@njit(parallel=True)
def update_item_biases_numba(
    N,
    mu,
    gamma,
    U,
    V,
    user_biases,
    movie_users_flat,
    movie_users_ptr,
    movie_users_len,
    movie_ratings_flat,
    item_biases,
):
    for n in prange(N):
        start = movie_users_ptr[n]
        ln = movie_users_len[n]
        num = 0.0
        den = gamma
        Vn = V[n]
        for k in range(ln):
            idx = start + k
            m = movie_users_flat[idx]
            r = movie_ratings_flat[idx]
            dot = 0.0
            Um = U[m]
            for t in range(Vn.shape[0]):
                dot += Um[t] * Vn[t]
            num += r - mu - user_biases[m] - dot
            den += 1.0
        item_biases[n] = num / den if den > 0.0 else 0.0



@njit(parallel=True)
def update_item_latent_numba(
    N,
    K,
    mu,
    tau,
    U,
    V,
    user_biases,
    item_biases,
    movie_users_flat,
    movie_users_ptr,
    movie_users_len,
    movie_ratings_flat,
):
    eyes = np.eye(K) * tau
    for n in prange(N):
        start = movie_users_ptr[n]
        ln = movie_users_len[n]
        if ln == 0:
            continue

        A = eyes.copy()
        b = np.zeros(K)
        ib = item_biases[n]

        for k in range(ln):
            idx = start + k
            m = movie_users_flat[idx]
            r = movie_ratings_flat[idx]
            Um = U[m]
            ub = user_biases[m]
            error = r - mu - ub - ib
            b += Um * error
            for i in range(K):
                for j in range(K):
                    A[i, j] += Um[i] * Um[j]
        V[n] = np.linalg.solve(A, b)


@njit(parallel=True)
def calc_rmse_numba(u_indices, m_indices, ratings, mu, U, V, user_biases, item_biases):
    sq_err = 0.0
    count = len(ratings)
    if count == 0:
        return 0.0

    for i in prange(count):
        u = u_indices[i]
        m = m_indices[i]
        r = ratings[i]
        dot = 0.0
        for k in range(U.shape[1]):
            dot += U[u, k] * V[m, k]
        pred = mu + user_biases[u] + item_biases[m] + dot
        sq_err += (r - pred) ** 2

    return np.sqrt(sq_err / count)


@njit(parallel=True)
def update_user_biases_numba(M, mu, lambd, gamma, U, V, item_biases,
                             user_items_flat, user_items_ptr, user_items_len,
                             user_ratings_flat, user_biases):
    for m in prange(M):
        start = user_items_ptr[m]
        ln = user_items_len[m]
        num = 0.0
        # According to Slide 69: num = lambda * sum(r - predictions_without_bu)
        for k in range(ln):
            idx = start + k
            n = user_items_flat[idx]
            r = user_ratings_flat[idx]
            
            # Predict excluding current user bias
            dot = 0.0
            Um = U[m]
            Vn = V[n]
            for t in range(Vn.shape[0]):
                dot += Um[t] * Vn[t]
            
            num += (r - mu - item_biases[n] - dot)
        
        # According to Slide 69: bu = (lambda * num) / (lambda * count + gamma)
        denominator = (lambd * ln) + gamma
        user_biases[m] = (lambd * num) / denominator if denominator > 0 else 0.0

@njit(parallel=True)
def update_user_latent_numba(M, K, mu, lambd, tau, U, V, user_biases, item_biases,
                            user_items_flat, user_items_ptr, user_items_len,
                            user_ratings_flat):
    for m in prange(M):
        start = user_items_ptr[m]
        ln = user_items_len[m]
        if ln == 0: 
            continue
        
        # According to Slide 71: A = (lambda * sum(V V.T)) + tau * I
        A = np.eye(K) * tau
        b = np.zeros(K)
        ub = user_biases[m]
        
        for k in range(ln):
            idx = start + k
            n = user_items_flat[idx]
            r = user_ratings_flat[idx]
            Vn = V[n]
            
            # Residual excluding current latent interaction
            error = r - mu - ub - item_biases[n]
            
            for t in range(K):
                # b = lambda * sum(V * error)
                b[t] += lambd * Vn[t] * error
                for p in range(K):
                    A[t, p] += lambd * Vn[t] * Vn[p]
        
        U[m] = np.linalg.solve(A, b)

@njit(parallel=True)
def calc_total_loss_numba(u_indices, m_indices, ratings, mu, U, V, user_biases, item_biases, lambd, tau, gamma):
    """
    Negative Log-Likelihood matching Slide 69:
    NLL = 0.5 * lambda * SSE + 0.5 * tau * sum(U^2 + V^2) + 0.5 * gamma * sum(biases^2)
    """
    sse = 0.0
    for i in prange(len(ratings)):
        u = u_indices[i]
        m = m_indices[i]
        r = ratings[i]
        dot = 0.0
        for k in range(U.shape[1]):
            dot += U[u, k] * V[m, k]
        pred = mu + user_biases[u] + item_biases[m] + dot
        sse += (r - pred) ** 2

    reg_vectors = 0.0
    for i in prange(U.shape[0]): 
        reg_vectors += np.sum(U[i]**2)
    for j in prange(V.shape[0]): 
        reg_vectors += np.sum(V[j]**2)

    reg_biases = np.sum(user_biases**2) + np.sum(item_biases**2)

    return (0.5 * lambd * sse) + (0.5 * tau * reg_vectors) + (0.5 * gamma * reg_biases)
class ALSRecommender:
    def __init__(
        self, fit_data, K=10, lambd=1.0,  tau=1, gamma=1, num_iters=10, test_ratio=0.2
    ):
        self.df = fit_data
        self.K = K
        self.lambd = lambd
        self.tau = tau
        self.gamma = gamma  # For biases
        self.num_iters = num_iters
        self.test_ratio = test_ratio

        self.M = self.df.num_users()
        self.N = self.df.num_movies()

        # Init Latent Factors
        self.U = np.random.randn(self.M, K) * 0.1
        self.V = np.random.randn(self.N, K) * 0.1
        self.user_biases = np.zeros(self.M)
        self.item_biases = np.zeros(self.N)
        self.mu = 0.0

        self.rmse_hist = []
        self.rmse_test_hist = []
        self.loss_hist = []

    def train_test_split(self):
        print("[LOG] Flattening FitData structure for Numba...")

        # Convert List-of-Lists (from FitData) to Flat Arrays (for Numba)
        train_u, train_i, train_r = [], [], []
        test_u, test_i, test_r = [], [], []

        for u_idx, ratings in enumerate(self.df.data_by_user):
            for movie_id, rating in ratings:
                m_idx = self.df.movie_id_to_idx[movie_id]
                if np.random.rand() < self.test_ratio:
                    test_u.append(u_idx)
                    test_i.append(m_idx)
                    test_r.append(rating)
                else:
                    train_u.append(u_idx)
                    train_i.append(m_idx)
                    train_r.append(rating)

        self.train_users = np.array(train_u, dtype=np.int32)
        self.train_items = np.array(train_i, dtype=np.int32)
        self.train_ratings = np.array(train_r, dtype=np.float64)

        self.test_users = np.array(test_u, dtype=np.int32)
        self.test_items = np.array(test_i, dtype=np.int32)
        self.test_ratings = np.array(test_r, dtype=np.float64)

        self.mu = np.mean(self.train_ratings) if len(self.train_ratings) > 0 else 0.0
        print(
            f"[LOG] Split complete. Train: {len(self.train_ratings)}, Test: {len(self.test_ratings)}"
        )

        # Build CSR-like structures
        print("[LOG] Building CSR index structures...")

        # User-centric
        sort_u = np.argsort(self.train_users)
        self.user_items_flat = self.train_items[sort_u]
        self.user_ratings_flat = self.train_ratings[sort_u]
        sorted_users = self.train_users[sort_u]
        self.user_items_ptr = np.zeros(self.M + 1, dtype=np.int32)
        counts_u = np.bincount(sorted_users, minlength=self.M)
        self.user_items_ptr[1:] = np.cumsum(counts_u)
        self.user_items_len = counts_u.astype(np.int32)

        # Item-centric
        sort_m = np.argsort(self.train_items)
        self.movie_users_flat = self.train_users[sort_m]
        self.movie_ratings_flat = self.train_ratings[sort_m]
        sorted_movies = self.train_items[sort_m]
        self.movie_users_ptr = np.zeros(self.N + 1, dtype=np.int32)
        counts_m = np.bincount(sorted_movies, minlength=self.N)
        self.movie_users_ptr[1:] = np.cumsum(counts_m)
        self.movie_users_len = counts_m.astype(np.int32)

    def fit(self):
        print(f"[LOG] Training ALS with K={self.K}...")
        for it in range(self.num_iters):
            t0 = time.time()
            # 1. Update User Biases
            update_user_biases_numba(
                self.M,
                self.mu,
                self.lambd,
                self.gamma,
                self.U,
                self.V,
                self.item_biases,
                self.user_items_flat,
                self.user_items_ptr,
                self.user_items_len,
                self.user_ratings_flat,
                self.user_biases,
            )
            # 2. Update Item Biases
            update_item_biases_numba(
                self.N,
                self.mu,
                self.lambd,
                self.gamma,
                self.U,
                self.V,
                self.user_biases,
                self.movie_users_flat,
                self.movie_users_ptr,
                self.movie_users_len,
                self.movie_ratings_flat,
                self.item_biases,
            )
            # 3. Update User Latent Vectors
            update_user_latent_numba(
                self.M,
                self.K,
                self.mu,
                self.lambd,
                self.tau,
                self.U,
                self.V,
                self.user_biases,
                self.item_biases,
                self.user_items_flat,
                self.user_items_ptr,
                self.user_items_len,
                self.user_ratings_flat,
            )
            # 4. Update Item Latent Vectors
            update_item_latent_numba(
                self.N,
                self.K,
                self.mu,
                self.lambd,
                self.tau,
                self.U,
                self.V,
                self.user_biases,
                self.item_biases,
                self.movie_users_flat,
                self.movie_users_ptr,
                self.movie_users_len,
                self.movie_ratings_flat,
            )

            loss = calc_total_loss_numba(
                self.train_users,
                self.train_items,
                self.train_ratings,
                self.mu,
                self.U,
                self.V,
                self.user_biases,
                self.item_biases,
                self.tau,
                self.gamma,
            )
            self.loss_hist.append(loss)

            # Metrics
            sample_size = min(100000, len(self.train_users))
            idx = np.random.choice(len(self.train_users), sample_size, replace=False)
            rmse_train = calc_rmse_numba(
                self.train_users[idx],
                self.train_items[idx],
                self.train_ratings[idx],
                self.mu,
                self.U,
                self.V,
                self.user_biases,
                self.item_biases,
            )
            rmse_test = calc_rmse_numba(
                self.test_users,
                self.test_items,
                self.test_ratings,
                self.mu,
                self.U,
                self.V,
                self.user_biases,
                self.item_biases,
            )

            self.rmse_hist.append(rmse_train)
            self.rmse_test_hist.append(rmse_test)
            print(
                f"  [Iter {it+1}] Time: {time.time()-t0:.1f}s | Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}"
            )

    def save_model(self, filename):
        # Temporarily remove heavy training data to save disk space
        temp_data = (
            self.train_users,
            self.train_items,
            self.train_ratings,
            self.test_users,
            self.test_items,
            self.test_ratings,
            self.user_items_flat,
            self.user_ratings_flat,
            self.movie_users_flat,
            self.movie_ratings_flat,
        )
        self.train_users = self.train_items = self.train_ratings = None
        self.test_users = self.test_items = self.test_ratings = None
        self.user_items_flat = self.user_ratings_flat = None
        self.movie_users_flat = self.movie_ratings_flat = None
        self.df = None  # Don't pickle the CSV loader data

        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"[LOG] Model saved to {filename}")

        # Restore data to continue execution if needed
        (
            self.train_users,
            self.train_items,
            self.train_ratings,
            self.test_users,
            self.test_items,
            self.test_ratings,
            self.user_items_flat,
            self.user_ratings_flat,
            self.movie_users_flat,
            self.movie_ratings_flat,
        ) = temp_data


def run_multi_k_training(fit_data, k_values=[2, 10, 20], num_iters=20):
    results = {}
    for K in k_values:
        print(
            f"\n================ STARTING K={K} (Iterations={num_iters}) ================"
        )

        # Pass num_iters here
        model = ALSRecommender(fit_data, K=K, tau=0.01, num_iters=num_iters)

        model.train_test_split()
        model.fit()

        filename = f"models/als_model_k{K}.pkl"
        model.save_model(filename)
        results[K] = model

    return results


def plot_comparisons(model_dict, save_path="figures/als_training_comparison_32m.pdf"):
    """
    Plots:
    (a) Training RMSE
    (b) Test RMSE
    (c) Negative Log-Likelihood (Total Loss)
    All in one row (3 per row) and saved as a single PDF.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # -----------------------------------
    # (a) TRAIN RMSE
    # -----------------------------------
    for K, model in model_dict.items():
        axes[0].plot(
            model.rmse_hist, marker="o", linestyle="--", alpha=0.8, label=f"K={K}"
        )

    axes[0].set_title("a. Training RMSE")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True)
    axes[0].legend()

    # -----------------------------------
    # (b) TEST RMSE
    # -----------------------------------
    for K, model in model_dict.items():
        axes[1].plot(
            model.rmse_test_hist, marker="o", linestyle="-", alpha=0.9, label=f"K={K}"
        )

    axes[1].set_title("b. Test RMSE")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True)
    axes[1].legend()

    # -----------------------------------
    # (c) LOG-LIKELIHOOD (NEGATIVE TOTAL LOSS)
    # -----------------------------------
    for K, model in model_dict.items():
        if not hasattr(model, "loss_hist"):
            continue  # safety check

        axes[2].plot(
            model.loss_hist, marker="s", linestyle="-", alpha=0.85, label=f"K={K}"
        )

    axes[2].set_title("c. Negative Log-Likelihood")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Total Loss")
    axes[2].grid(True)
    axes[2].legend()

    # -----------------------------------
    # Layout & Save
    # -----------------------------------
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"[LOG] Plot saved to {save_path}")

    plt.close(fig)


def load_and_test_pickle(filename, fit_data_obj):
    if not os.path.exists(filename):
        print(f"[ERROR] File {filename} not found.")
        return

    print(f"\n[LOG] Loading {filename}...")
    with open(filename, "rb") as f:
        model = pickle.load(f)

    print(f"  > Loaded Model K={model.K}, tau={model.tau}")
    print(f"  > Final Test RMSE stored: {model.rmse_test_hist[-1]:.4f}")

    # Sanity Check Prediction
    if len(fit_data_obj.idx_to_user_id) > 0:
        # 1. Get a real user ID
        real_user_id = fit_data_obj.idx_to_user_id[0]
        u_idx = fit_data_obj.user_id_to_idx[real_user_id]

        print(f"  > Predictions for User {real_user_id}:")

        # 2. Pick 5 random internal movie indices
        random_indices = np.random.randint(0, model.N, 5)

        # 3. Convert internal indices -> Real Movie IDs
        real_movie_ids = [fit_data_obj.idx_to_movie_id[i] for i in random_indices]

        # 4. Get Titles (The Class in Cell 2 has this method)
        titles = fit_data_obj.movie_titles(real_movie_ids)

        for i, m_idx in enumerate(random_indices):
            title = titles[i]
            # Predict
            pred = (
                model.mu
                + model.user_biases[u_idx]
                + model.item_biases[m_idx]
                + np.dot(model.U[u_idx], model.V[m_idx])
            )

            print(f"    Movie: '{title}' -> Pred Rating: {pred:.2f}")


def get_recommendations(model, user_idx, exclude_indices=set(), top_n=10):
    """
    Calculates scores for all items for a specific user and returns top N.
    """
    # Score = mu + b_u + b_i + U . V
    # Calculate dot product for this user against ALL items
    scores = (
        model.mu
        + model.user_biases[user_idx]
        + model.item_biases
        + (model.V @ model.U[user_idx])
    )

    # Set scores of already rated items to negative infinity so they aren't recommended
    for m_idx in exclude_indices:
        scores[m_idx] = -np.inf

    # Get top N indices
    # argsort gives ascending, so we take last N and reverse
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return top_indices


def get_polarizing_movies(model, top_n=10):
    """
    Returns indices of movies with the largest vector norms.
    Large norm = strong latent traits (loved by some, hated by others).
    """
    # Calculate Euclidean norm for every movie vector in V
    norms = np.linalg.norm(model.V, axis=1)

    # Get indices of largest norms
    top_indices = np.argsort(norms)[-top_n:][::-1]

    return list(zip(top_indices, norms[top_indices]))


def optimize_single_user(model, user_idx, rated_item_indices, ratings, num_iters=10):
    """
    Solves the ALS equation for a SINGLE user to update their vector U[user_idx]
    based on new ratings.
    """
    K = model.K
    tau = model.tau

    # Slice the V matrix to get only the movies the user rated
    V_batch = model.V[rated_item_indices]  # shape: (n_rated, K)
    item_biases_batch = model.item_biases[rated_item_indices]

    # Pre-calculate V^T * V + tau * I (This is the 'A' matrix in Ax=b)
    # Note: For a single user with few ratings, we can do this simply
    A = np.dot(V_batch.T, V_batch) + np.eye(K) * tau

    for _ in range(num_iters):
        # Update User Bias
        # b_u = sum(r - mu - b_i - U.V) / (n_ratings + tau)
        pred_ratings = np.dot(V_batch, model.U[user_idx])
        errors = ratings - model.mu - item_biases_batch - pred_ratings
        model.user_biases[user_idx] = np.sum(errors + model.user_biases[user_idx]) / (
            len(ratings) + tau
        )

        # Update User Latent Vector (U)
        # b = V^T * (ratings - mu - b_u - b_i)
        residual = ratings - model.mu - model.user_biases[user_idx] - item_biases_batch
        b = np.dot(V_batch.T, residual)

        # Solve Ax = b
        model.U[user_idx] = np.linalg.solve(A, b)

def run_hyperparameter_search(fit_data):
    """
    Prints and stores all combinations of K, tau, and gamma.
    """
    k_values = [2, 10, 20]
    tau_values = [0.01, 0.1, 1.0]
    gamma_values = [0.01, 0.1, 1.0]
    num_iters = 30
    
    search_results = []
    print(f"{'K':<4} | {'tau':<5} | {'gamma':<5} | {'Train RMSE':<10} | {'Test RMSE':<10} | {'Total Loss (NLL)':<15}")
    print("-" * 75)

    for K in k_values:
        for T in tau_values:
            for G in gamma_values:
                model = ALSRecommender(fit_data, K=K, tau=T, gamma=G, num_iters=num_iters)
                model.train_test_split()
                model.fit()
                
                final_loss = model.loss_hist[-1]
                
                res = {
                    'K': K, 'tau': T, 'gamma': G,
                    'Train_RMSE': round(model.rmse_hist[-1], 4),
                    'Test_RMSE': round(model.rmse_test_hist[-1], 4),
                    'Total_Loss': round(final_loss, 2)
                }
                search_results.append(res)
                print(f"{K:<4} | {T:<5} | {G:<5} | {res['Train_RMSE']:<10} | {res['Test_RMSE']:<10} | {res['Total_Loss']:<15,}")

    return pd.DataFrame(search_results)

# Execute
# results_table = run_full_grid_search(fit_data)