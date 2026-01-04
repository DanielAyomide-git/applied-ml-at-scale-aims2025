import csv
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, prange

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





# --------------------------
# NUMBA OPTIMIZED UPDATES
# --------------------------

# --------------------------
# Numba-optimized update functions
# --------------------------
@njit(parallel=True)
def update_user_biases_numba(M, mu, gamma, U, V, item_biases,
                             user_items_flat, user_items_ptr, user_items_len,
                             user_ratings_flat, user_biases):
    for m in prange(M):
        start = user_items_ptr[m]
        ln = user_items_len[m]
        num = 0.0
        den = gamma  
        Um = U[m]
        for k in range(ln):
            idx = start + k
            n = user_items_flat[idx]
            r = user_ratings_flat[idx]
            dot = 0.0
            Vn = V[n]
            for t in range(Vn.shape[0]):
                dot += Um[t] * Vn[t]
            num += r - mu - item_biases[n] - dot
            den += 1.0
        user_biases[m] = num / den if den > 0.0 else 0.0

@njit(parallel=True)
def update_item_biases_numba(N, mu, gamma, U, V, user_biases,
                             movie_users_flat, movie_users_ptr, movie_users_len,
                             movie_ratings_flat, item_biases):
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
def update_user_latent_numba(M, K, mu, tau, U, V, user_biases, item_biases,
                            user_items_flat, user_items_ptr, user_items_len,
                            user_ratings_flat):
    # Precompute V.T @ V for the entire dataset once to speed up "all-item" calculations
    # This is a trick used in "Implicit ALS" but can be adapted.
    
    for m in prange(M):
        start = user_items_ptr[m]
        ln = user_items_len[m]
        if ln == 0:
             continue
        
        # A = V_batch.T @ V_batch + tau * I
        # b = V_batch.T @ (ratings - offset)
        A = np.eye(K) * tau
        b = np.zeros(K)
        ub = user_biases[m]
        
        for k in range(ln):
            idx = start + k
            n = user_items_flat[idx]
            r = user_ratings_flat[idx]
            Vn = V[n]
            # Use dot product for b
            error = r - mu - ub - item_biases[n]
            for t in range(K):
                b[t] += Vn[t] * error
                # Rank-1 update of A
                for p in range(K):
                    A[t, p] += Vn[t] * Vn[p]
        
        # Solving the small KxK system
        U[m] = np.linalg.solve(A, b)

@njit(parallel=True)
def update_item_latent_numba(N, K, mu, tau, U, V, user_biases, item_biases,
                             movie_users_flat, movie_users_ptr, movie_users_len,
                             movie_ratings_flat):
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
                    A[i,j] += Um[i] * Um[j]
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
        sq_err += (r - pred)**2
        
    return np.sqrt(sq_err / count)


@njit(parallel=True)
def calc_total_loss_numba(u_indices, m_indices, ratings, mu, U, V, user_biases, item_biases, tau, gamma):
    """
    Calculates the Total Loss (Negative Log-Likelihood) based on Slide 158.
    
    Formula: 
    Loss = 0.5 * SSE + 0.5 * tau * (sum(U^2) + sum(V^2)) + 0.5 * gamma * (sum(bu^2) + sum(bi^2))
    """
    # 1. Calculate Sum of Squared Errors (SSE)
    # This represents the Data Likelihood term
    sse = 0.0
    for i in prange(len(ratings)):
        u = u_indices[i]
        m = m_indices[i]
        r = ratings[i]
        
        # Compute dot product U_m . V_n
        dot = 0.0
        for k in range(U.shape[1]):
            dot += U[u, k] * V[m, k]
            
        # Prediction: mu + b_u + b_i + (U . V)
        pred = mu + user_biases[u] + item_biases[m] + dot
        sse += (r - pred)**2
    
    # 2. Calculate Latent Vector Regularization (tau)
    reg_u = 0.0
    for i in prange(U.shape[0]):
        for k in range(U.shape[1]):
            reg_u += U[i, k]**2
            
    reg_v = 0.0
    for j in prange(V.shape[0]):
        for k in range(V.shape[1]):
            reg_v += V[j, k]**2

    # 3. Calculate Bias Regularization (gamma)
    reg_bu = 0.0
    for i in prange(user_biases.shape[0]):
        reg_bu += user_biases[i]**2
        
    reg_bi = 0.0
    for j in prange(item_biases.shape[0]):
        reg_bi += item_biases[j]**2

    # Final Combination according to the probabilistic derivation
    # total_loss = (Likelihood) + (Prior on Vectors) + (Prior on Biases)
    total_loss = (0.5 * sse) + (0.5 * tau * (reg_u + reg_v)) + (0.5 * gamma * (reg_bu + reg_bi))
    
    return total_loss

class ALSRecommender:
    def __init__(self, fit_data, K=10, tau=0.1, gamma=0.01, num_iters=10, test_ratio=0.2):
        self.df = fit_data
        self.K = K
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

    def train_test_split(self):
        print("[LOG] Flattening FitData structure for Numba...")
        
        # Convert List-of-Lists (from FitData) to Flat Arrays (for Numba)
        train_u, train_i, train_r = [], [], []
        test_u, test_i, test_r = [], [], []
        
        for u_idx, ratings in enumerate(self.df.data_by_user):
            for (movie_id, rating) in ratings:
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
        print(f"[LOG] Split complete. Train: {len(self.train_ratings)}, Test: {len(self.test_ratings)}")
        
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
                self.M, self.mu, self.gamma, self.U, self.V, self.item_biases,
                self.user_items_flat, self.user_items_ptr, self.user_items_len,
                self.user_ratings_flat, self.user_biases
            )
            # 2. Update Item Biases
            update_item_biases_numba(
                self.N, self.mu, self.gamma, self.U, self.V, self.user_biases,
                self.movie_users_flat, self.movie_users_ptr, self.movie_users_len,
                self.movie_ratings_flat, self.item_biases
            )
            # 3. Update User Latent Vectors
            update_user_latent_numba(
                self.M, self.K, self.mu, self.tau, self.U, self.V, self.user_biases, self.item_biases,
                self.user_items_flat, self.user_items_ptr, self.user_items_len, self.user_ratings_flat
            )
            # 4. Update Item Latent Vectors
            update_item_latent_numba(
                self.N, self.K, self.mu, self.tau, self.U, self.V, self.user_biases, self.item_biases,
                self.movie_users_flat, self.movie_users_ptr, self.movie_users_len, self.movie_ratings_flat
            )
            
            # Metrics
            sample_size = min(100000, len(self.train_users))
            idx = np.random.choice(len(self.train_users), sample_size, replace=False)
            rmse_train = calc_rmse_numba(self.train_users[idx], self.train_items[idx], self.train_ratings[idx],
                                         self.mu, self.U, self.V, self.user_biases, self.item_biases)
            rmse_test = calc_rmse_numba(self.test_users, self.test_items, self.test_ratings,
                                        self.mu, self.U, self.V, self.user_biases, self.item_biases)
            
            self.rmse_hist.append(rmse_train)
            self.rmse_test_hist.append(rmse_test)
            print(f"  [Iter {it+1}] Time: {time.time()-t0:.1f}s | Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")

    def save_model(self, filename):
        # Temporarily remove heavy training data to save disk space
        temp_data = (self.train_users, self.train_items, self.train_ratings,
                     self.test_users, self.test_items, self.test_ratings,
                     self.user_items_flat, self.user_ratings_flat,
                     self.movie_users_flat, self.movie_ratings_flat)
        self.train_users = self.train_items = self.train_ratings = None
        self.test_users = self.test_items = self.test_ratings = None
        self.user_items_flat = self.user_ratings_flat = None
        self.movie_users_flat = self.movie_ratings_flat = None
        self.df = None # Don't pickle the CSV loader data
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"[LOG] Model saved to {filename}")
        
        # Restore data to continue execution if needed
        (self.train_users, self.train_items, self.train_ratings,
         self.test_users, self.test_items, self.test_ratings,
         self.user_items_flat, self.user_ratings_flat,
         self.movie_users_flat, self.movie_ratings_flat) = temp_data




def run_multi_k_training(fit_data, k_values=[2, 10, 20], num_iters=20):
    results = {}
    for K in k_values:
        print(f"\n================ STARTING K={K} (Iterations={num_iters}) ================")
        
        # Pass num_iters here
        model = ALSRecommender(fit_data, K=K, tau=0.01, num_iters=num_iters) 
        
        model.train_test_split()
        model.fit()
        
        filename = f"models/als_model_k{K}.pkl"
        model.save_model(filename)
        results[K] = model 
        
    return results


def plot_comparisons(model_dict, save_path="figures/als_training_comparison_32m.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # -----------------------------------
    # LEFT: TRAIN RMSE
    # -----------------------------------
    for K, model in model_dict.items():
        axes[0].plot(
            model.rmse_hist,
            marker='o',
            linestyle='--',
            alpha=0.8,
            label=f'K={K}'
        )

    axes[0].set_title("Training RMSE")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True)
    axes[0].legend()

    # -----------------------------------
    # RIGHT: TEST RMSE
    # -----------------------------------
    for K, model in model_dict.items():
        axes[1].plot(
            model.rmse_test_hist,
            marker='o',
            linestyle='-',
            alpha=0.9,
            label=f'K={K}'
        )

    axes[1].set_title("Test RMSE")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True)
    axes[1].legend()

    # -----------------------------------
    # Overall formatting
    # -----------------------------------
    fig.suptitle("ALS Training vs Test RMSE (32M Dataset)", fontsize=14)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save both plots in ONE PDF
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"[LOG] Plot saved to {save_path}")
    
    
    plt.close("all")

    plt.show()

def load_and_test_pickle(filename, fit_data_obj):
    if not os.path.exists(filename):
        print(f"[ERROR] File {filename} not found.")
        return

    print(f"\n[LOG] Loading {filename}...")
    with open(filename, 'rb') as f:
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
            pred = model.mu + model.user_biases[u_idx] + model.item_biases[m_idx] + \
                   np.dot(model.U[u_idx], model.V[m_idx])
            
            print(f"    Movie: '{title}' -> Pred Rating: {pred:.2f}")


def get_recommendations(model, user_idx, exclude_indices=set(), top_n=10):
    """
    Calculates scores for all items for a specific user and returns top N.
    """
    # Score = mu + b_u + b_i + U . V
    # Calculate dot product for this user against ALL items
    scores = model.mu + model.user_biases[user_idx] + model.item_biases + \
             (model.V @ model.U[user_idx])
    
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
        model.user_biases[user_idx] = np.sum(errors + model.user_biases[user_idx]) / (len(ratings) + tau)
        
        # Update User Latent Vector (U)
        # b = V^T * (ratings - mu - b_u - b_i)
        residual = ratings - model.mu - model.user_biases[user_idx] - item_biases_batch
        b = np.dot(V_batch.T, residual)
        
        # Solve Ax = b
        model.U[user_idx] = np.linalg.solve(A, b)

def run_hyperparameter_search(fit_data):
    # Search over K and tau; keeping gamma = 0.1 for stability
    k_values = [2, 10, 20]
    tau_values = [0.01, 0.1, 1.0]
    fixed_gamma = 0.01
    num_iters = 10 
    
    search_results = []
    print(f"{'K':<5} | {'tau':<8} | {'Train RMSE':<12} | {'Test RMSE':<12} | {'Loss':<15}")
    print("-" * 60)

    for K in k_values:
        for T in tau_values:
            model = ALSRecommender(fit_data, K=K, tau=T, gamma=fixed_gamma, num_iters=num_iters)
            model.train_test_split()
            model.fit()
            
            final_train_rmse = model.rmse_hist[-1]
            final_test_rmse = model.rmse_test_hist[-1]
            
            # Use the full loss function with both tau and gamma
            final_loss = calc_total_loss_numba(
                model.train_users, model.train_items, model.train_ratings,
                model.mu, model.U, model.V, model.user_biases, model.item_biases, 
                model.tau, model.gamma
            )
            
            res = {
                'K': K,
                'tau': T,
                'Train_RMSE': round(final_train_rmse, 4),
                'Test_RMSE': round(final_test_rmse, 4),
                'Total_Loss': round(final_loss, 2)
            }
            search_results.append(res)
            print(f"{K:<5} | {T:<8} | {res['Train_RMSE']:<12} | {res['Test_RMSE']:<12} | {res['Total_Loss']:<15}")

    return pd.DataFrame(search_results)