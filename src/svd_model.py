import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, lil_matrix
import os

class SVDRecommender:
    """
    Collaborative Filtering using Mean-Centered TruncatedSVD (sklearn).
    Learns latent factors between Users and Books based on the Ratings matrix.

    KEY IMPROVEMENT — Mean Centering:
      Raw TruncatedSVD treats missing entries as 0, which biases predictions
      towards 0 instead of the 1-5 scale.  By subtracting each user's mean
      rating BEFORE decomposition, missing entries effectively become
      "no information" (0 = neutral).  After reconstruction we add the
      user mean back to restore the original rating scale.

    WHY TruncatedSVD: Works on Sparse matrices without needing C++ compilation.
    """
    def __init__(self, n_factors=50):
        self.n_factors        = n_factors
        self.model            = TruncatedSVD(n_components=n_factors, random_state=42)
        self.books_df         = None
        self.user_index       = {}   # user_id  -> row index in sparse matrix
        self.book_index       = {}   # book_id  -> col index in sparse matrix
        self.predicted_matrix = None  # Full dense predicted matrix (users x books)
        self.user_means       = None  # Per-user mean rating (for mean-centering)
        self.rated_books      = {}   # user_id -> set of book_ids they already rated
        self.global_mean      = 0.0  # Fallback for unknown users

    def fit(self, ratings_df, books_df):
        """
        Train the SVD model by decomposing the Mean-Centered User-Item matrix.
        """
        self.books_df = books_df

        # Build index mappings
        users = ratings_df['user_id'].unique()
        books = ratings_df['book_id'].unique()
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.book_index = {bid: i for i, bid in enumerate(books)}

        n_users = len(users)
        n_books = len(books)

        # Store which books each user has already rated (for filtering)
        self.rated_books = ratings_df.groupby('user_id')['book_id'].apply(set).to_dict()

        # Build sparse ratings matrix (rows=users, cols=books)
        print("Building User-Item Matrix...")
        rows = ratings_df['user_id'].map(self.user_index).values
        cols = ratings_df['book_id'].map(self.book_index).values
        vals = ratings_df['rating'].astype(float).values
        sparse_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_books))
        print(f"Matrix shape: {sparse_matrix.shape} (users × books)")

        # ── Mean-Centering ─────────────────────────────────────────
        # Compute per-user mean rating (only over actually rated entries)
        print("Mean-centering ratings (per user)...")
        self.global_mean = float(ratings_df['rating'].mean())
        user_sum   = np.array(sparse_matrix.sum(axis=1)).flatten()      # sum of ratings per user
        user_count = np.array((sparse_matrix != 0).sum(axis=1)).flatten()  # number of ratings
        user_count[user_count == 0] = 1   # avoid division by zero
        self.user_means = user_sum / user_count                          # shape: (n_users,)

        # Subtract user mean from each rated entry (only non-zero entries)
        centered = lil_matrix(sparse_matrix.shape, dtype=np.float64)
        cx = sparse_matrix.tocoo()
        for r, c, v in zip(cx.row, cx.col, cx.data):
            centered[r, c] = v - self.user_means[r]
        centered = centered.tocsr()

        # ── SVD Decomposition ──────────────────────────────────────
        print(f"Applying TruncatedSVD with {self.n_factors} factors...")
        U  = self.model.fit_transform(centered)    # shape: users × k
        VT = self.model.components_                # shape: k × books
        approx = np.dot(U, VT)                     # shape: users × books (centered)

        # Add user means back → predictions on original 1-5 scale
        self.predicted_matrix = approx + self.user_means[:, np.newaxis]
        print("SVD Model Trained Successfully!")

    def predict_rating(self, user_id, book_id):
        """Predict the rating a user would give to a specific book."""
        if user_id not in self.user_index or book_id not in self.book_index:
            return self.global_mean
        u = self.user_index[user_id]
        b = self.book_index[book_id]
        raw = self.predicted_matrix[u, b]
        return float(np.clip(raw, 1, 5))

    def get_user_recommendations(self, user_id, top_k=10):
        """
        Generate Top K recommendations for a specific User.
        Filters out books already rated by the user.
        """
        if user_id not in self.user_index:
            print(f"User {user_id} not found in training data (Cold Start).")
            return pd.DataFrame()

        u_idx = self.user_index[user_id]
        already_rated = self.rated_books.get(user_id, set())

        # Build list of (book_id, predicted_score) — exclude already rated
        predictions = []
        for book_id, b_idx in self.book_index.items():
            if book_id in already_rated:
                continue
            score = np.clip(self.predicted_matrix[u_idx, b_idx], 1, 5)
            predictions.append((book_id, float(score)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_book_ids   = [p[0] for p in predictions[:top_k]]
        predicted_vals = [p[1] for p in predictions[:top_k]]

        recs = self.books_df[self.books_df['book_id'].isin(top_book_ids)].copy()
        score_map = dict(zip(top_book_ids, predicted_vals))
        recs['predicted_rating'] = recs['book_id'].map(score_map)
        recs = recs.sort_values('predicted_rating', ascending=False).reset_index(drop=True)

        return recs[['book_id', 'title', 'authors', 'genres', 'predicted_rating']]
