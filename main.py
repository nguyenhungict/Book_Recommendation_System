"""
main.py — Entry point for the Book Recommendation System demo.
Run: python main.py
     .\venv\Scripts\python main.py  (if using venv on Windows)
"""
import os
import sys

from src.preprocessing import load_and_clean_data, create_feature_matrix
from src.knn_model import KNNRecommender
from src.svd_model import SVDRecommender
from src.hybrid import HybridRecommender

# ── CONFIG ──────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
TOP_K      = 10

# Pre-selected demo users (from project plan)
DEMO_USERS = {
    "heavy":    None,   # Will be auto-selected: user with 50+ ratings
    "light":    None,   # Will be auto-selected: user with < 5 ratings
    "genre":    None,   # Will be auto-selected: user who rates mostly one genre
}
# ────────────────────────────────────────────────────────


def load_or_train(books, ratings, feature_matrix):
    """Load saved models if they exist; otherwise train from scratch."""
    knn = KNNRecommender()
    svd = SVDRecommender()

    knn_path = os.path.join(MODEL_DIR, "knn_model.pkl")
    svd_path = os.path.join(MODEL_DIR, "svd_model.pkl")

    if os.path.exists(knn_path):
        print("Loading saved KNN model...")
        knn.books_df = books
        knn.feature_matrix = feature_matrix
        knn.load_model(knn_path)
        from sklearn.neighbors import NearestNeighbors
    else:
        knn.fit(books, feature_matrix)
        knn.save_model(knn_path)

    # SVD always needs full predicted_matrix in memory; retrain if not saved
    print("Training SVD Model...")
    svd.fit(ratings, books)

    return knn, svd


def find_demo_users(ratings):
    """Auto-select 3 representative demo users."""
    rating_counts = ratings['user_id'].value_counts()

    # Heavy user: 50+ ratings
    heavy = rating_counts[rating_counts >= 50].index[0]

    # Light user: less than 5 ratings
    light = rating_counts[rating_counts < 5].index[0]

    # Genre-focused user: find user whose top-rated books are in one genre
    # (simplified: just pick a user with 10-20 ratings for demo purposes)
    genre_focused = rating_counts[(rating_counts >= 10) & (rating_counts <= 20)].index[0]

    return heavy, light, genre_focused


def print_recommendations(results, label=""):
    """Pretty-print recommendations."""
    print(f"\n{'─'*60}")
    print(f"  📚 {label}")
    print(f"{'─'*60}")
    if results is None or (hasattr(results, 'empty') and results.empty):
        print("  No recommendations found.")
        return
    for i, row in enumerate(results.itertuples(), 1):
        title   = getattr(row, 'title',   'N/A')
        authors = getattr(row, 'authors', 'N/A')
        rating  = getattr(row, 'average_rating', getattr(row, 'predicted_rating', 'N/A'))
        print(f"  {i:>2}. {title[:50]:<50} | {str(authors)[:25]:<25} | ★ {rating}")


def run_demo(hybrid, ratings):
    """Run the 3-scenario demo for presentation."""
    heavy_id, light_id, genre_id = find_demo_users(ratings)

    print("\n" + "═"*60)
    print("  BOOK RECOMMENDATION SYSTEM — DEMO")
    print("═"*60)

    # Demo 1: Cold Start user (no ratings) — KNN only
    print("\n[DEMO 1] Cold Start — Guest User (No Rating History)")
    print("  Input: Book title = 'The Hunger Games'")
    res = hybrid.recommend(book_title="The Hunger Games")
    print_recommendations(res, "Recommended (Content-Based KNN)")

    # Demo 2: Light user (< 5 ratings) — KNN fallback
    print(f"\n[DEMO 2] Light User (user_id={light_id}, <5 ratings)")
    light_count = ratings[ratings['user_id'] == light_id].shape[0]
    print(f"  This user has only {light_count} rating(s). KNN fallback applies.")
    res = hybrid.recommend(book_title="Twilight", user_id=light_id, ratings_df=ratings)
    print_recommendations(res, f"Recommended for user {light_id} (KNN Fallback)")

    # Demo 3: Heavy user (50+ ratings) — Full Hybrid
    print(f"\n[DEMO 3] Experienced User (user_id={heavy_id}, 50+ ratings)")
    heavy_count = ratings[ratings['user_id'] == heavy_id].shape[0]
    print(f"  This user has {heavy_count} ratings. Hybrid (KNN + SVD) applies.")
    res = hybrid.recommend(book_title="Harry Potter", user_id=heavy_id, ratings_df=ratings)
    print_recommendations(res, f"Recommended for user {heavy_id} (Hybrid KNN+SVD)")


def interactive_cli(hybrid, ratings):
    """Simple CLI loop for manual testing."""
    print("\n" + "═"*60)
    print("  INTERACTIVE MODE  (type 'quit' to exit)")
    print("═"*60)
    while True:
        print()
        book_title = input("Enter a book title (or press Enter to skip): ").strip()
        user_id_str = input("Enter your user_id  (or press Enter to skip): ").strip()

        if book_title.lower() == 'quit' or user_id_str.lower() == 'quit':
            print("Exiting. Goodbye!")
            break

        user_id = int(user_id_str) if user_id_str.isdigit() else None
        title   = book_title if book_title else None

        results = hybrid.recommend(book_title=title, user_id=user_id, ratings_df=ratings)
        print_recommendations(results, "Your Recommendations")


def main():
    print("Loading data...")
    books, ratings, tags = load_and_clean_data(
        os.path.join(DATA_DIR, "books.csv"),
        os.path.join(DATA_DIR, "ratings.csv"),
        os.path.join(DATA_DIR, "tags.csv"),
    )

    print("Building feature matrix...")
    feature_matrix, tfidf, scaler = create_feature_matrix(books)

    print("Loading / Training models...")
    knn, svd = load_or_train(books, ratings, feature_matrix)

    # Assemble Hybrid
    hybrid = HybridRecommender(knn, svd, books, feature_matrix)

    # Run demo
    run_demo(hybrid, ratings)

    # Then allow interactive input
    interactive_cli(hybrid, ratings)


if __name__ == "__main__":
    main()
