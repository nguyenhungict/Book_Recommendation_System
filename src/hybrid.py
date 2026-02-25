import pandas as pd

class HybridRecommender:
    """
    Hybrid Recommendation System that combines:
    - Content-Based KNN: for cold-start users OR book-to-book similarity
    - Collaborative Filtering SVD: for users with rating history
    WHY Hybrid: No single model is perfect. KNN handles new users; SVD handles personalization.
    """
    def __init__(self, knn_model, svd_model, books_df, feature_matrix,
                 knn_weight=0.4, svd_weight=0.6, cold_start_threshold=5):
        self.knn = knn_model
        self.svd = svd_model
        self.books_df = books_df
        self.feature_matrix = feature_matrix
        
        # Blend weights: SVD is trusted more when user has sufficient history
        self.knn_weight = knn_weight
        self.svd_weight = svd_weight
        
        # Users with fewer than this many ratings are considered "new" (Cold Start)
        # WHY 20: goodbooks dataset users all have at least a few ratings, so threshold is higher
        self.cold_start_threshold = cold_start_threshold

    def _is_cold_start(self, user_id, ratings_df):
        """
        Check if a user has too few ratings to be trusted by SVD.
        WHY: SVD needs a minimum history to find latent patterns.
        """
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        return len(user_ratings) < self.cold_start_threshold

    def recommend(self, book_title=None, user_id=None, ratings_df=None, top_k=10):
        """
        Main recommendation function. Chooses strategy based on available inputs.

        CASES:
        1. Only book_title  → Content-Based KNN only (Cold Start / guest user)
        2. Only user_id     → SVD only (experienced user, no pivot book)
        3. Both provided    → Hybrid: blend KNN + SVD scores (best results)
        """
        
        # --- CASE 1: No user_id → KNN only (cold start / guest) ---
        if user_id is None or ratings_df is None:
            if book_title is None:
                raise ValueError("Must provide at least a book_title or user_id.")
            print(f"[Strategy] COLD START — Using KNN Content-Based only.")
            return self.knn.get_recommendations(book_title)

        # --- Check if user is "old" or "new" ---
        is_cold = self._is_cold_start(user_id, ratings_df)

        # --- CASE 2: New user, only book lookup possible ---
        if is_cold:
            print(f"[Strategy] USER {user_id} has few ratings — KNN Fallback (Cold Start).")
            if book_title:
                return self.knn.get_recommendations(book_title)
            else:
                print("No book title provided for cold-start user. Please provide a book title.")
                return pd.DataFrame()

        # --- CASE 3: Experienced user + book_title → Full Hybrid ---
        print(f"[Strategy] USER {user_id} has rating history — Using Hybrid (KNN + SVD).")

        # Get KNN recommendations (book-to-book similarity)
        knn_results = self.knn.get_recommendations(book_title) if book_title else pd.DataFrame()

        # Get SVD recommendations (user preference-based)
        svd_results = self.svd.get_user_recommendations(user_id, top_k=top_k * 2)

        if svd_results.empty:
            return knn_results
        if knn_results is None or knn_results.empty:
            return svd_results.head(top_k)

        # --- Merge and Blend Scores ---
        # Normalize SVD predicted_rating to [0, 1] scale (original scale: 1-5)
        svd_results = svd_results.copy()
        svd_results['svd_score'] = (svd_results['predicted_rating'] - 1) / 4.0

        # KNN already has similarity_score in [0, 1]
        knn_results = knn_results.copy()
        knn_results['knn_score'] = knn_results['similarity_score']

        # Merge on title (books appearing in both lists get a blended score)
        merged = pd.merge(
            knn_results[['title', 'knn_score']],
            svd_results[['title', 'svd_score']],
            on='title', how='outer'
        ).fillna(0)

        # Bring back the book metadata (authors, genres, average_rating)
        meta = self.books_df[['title', 'authors', 'genres', 'average_rating']].drop_duplicates(subset=['title'])
        merged = pd.merge(merged, meta, on='title', how='left')

        # Weighted blend: final_score = 0.4 * KNN + 0.6 * SVD
        merged['final_score'] = (self.knn_weight * merged['knn_score'] +
                                 self.svd_weight * merged['svd_score'])

        merged = merged.sort_values('final_score', ascending=False).head(top_k)
        merged.reset_index(drop=True, inplace=True)

        return merged[['title', 'authors', 'genres', 'average_rating', 'final_score']]


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.preprocessing import load_and_clean_data, create_feature_matrix
    from src.knn_model import KNNRecommender
    from src.svd_model import SVDRecommender

    # ── Load Data ──────────────────────────────────────
    books, ratings, tags = load_and_clean_data(
        os.path.join('data', 'books.csv'),
        os.path.join('data', 'ratings.csv'),
        os.path.join('data', 'tags.csv')
    )
    feature_matrix, tfidf, scaler = create_feature_matrix(books)

    # ── Train Models ───────────────────────────────────
    knn = KNNRecommender(); knn.fit(books, feature_matrix)
    svd = SVDRecommender(); svd.fit(ratings, books)

    # ── Build Hybrid ───────────────────────────────────
    hybrid = HybridRecommender(knn, svd, books, feature_matrix)

    # ── Find demo users ────────────────────────────────
    counts = ratings['user_id'].value_counts()
    heavy_user = counts[counts >= 200].index[0]

    # goodbooks dataset: all users have many ratings, so "light" = fewest ratings in dataset
    light_user = counts.index[-1]   # user with the least ratings in the dataset
    light_count = counts.iloc[-1]
    heavy_count = counts[counts.index == heavy_user].iloc[0]

    print("\n" + "═"*65)
    print("  HYBRID SYSTEM TEST — 3 DEMO SCENARIOS")
    print("═"*65)

    def show(df, label):
        """Print results regardless of which score column is present."""
        print(f"\n  → {label}")
        if df is None or (hasattr(df, 'empty') and df.empty):
            print("  No results."); return
        score_col = 'final_score' if 'final_score' in df.columns else 'similarity_score'
        cols = ['title', 'average_rating', score_col]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(5).to_string(index=False))

    # CASE 1: Cold Start — no user_id at all
    print("\n[CASE 1] Cold Start — Guest user, book title only")
    res1 = hybrid.recommend(book_title="The Hunger Games")
    show(res1, "KNN Content-Based")

    # CASE 2: Force cold-start demo by creating a Hybrid with high threshold
    # WHY: goodbooks users all have many ratings; we simulate a "light" user
    #      by setting threshold=9999 so ANY user triggers KNN fallback
    demo_hybrid_cold = HybridRecommender(knn, svd, books, feature_matrix,
                                         cold_start_threshold=9999)
    print(f"\n[CASE 2] Light User (id={light_user}, {light_count} ratings) — KNN Fallback")
    print(f"         [cold_start_threshold set to 9999 to simulate cold start]")
    res2 = demo_hybrid_cold.recommend(book_title="Twilight", user_id=light_user, ratings_df=ratings)
    show(res2, "KNN Fallback (Cold Start simulated)")

    # CASE 3: Heavy user — Full Hybrid (real weighted blend)
    print(f"\n[CASE 3] Heavy User (id={heavy_user}, {heavy_count} ratings) — Full Hybrid")
    res3 = hybrid.recommend(book_title="Harry Potter", user_id=heavy_user, ratings_df=ratings)
    show(res3, "Hybrid KNN + SVD (blended score)")

    print("\n✅ Hybrid system working correctly!")

