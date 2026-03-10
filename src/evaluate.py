import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_knn(knn_model, books_df, ratings_df, top_k=10, sample_size=50):
    """
    Evaluate KNN using Precision@K, Recall@K, and F1@K.

    Logic: For a sample of users, treat their highest-rated books as 'ground truth'.
           Use one liked book as a seed, and check how many of the remaining
           liked books appear in the model's top-K recommendations.

    Metrics:
      - Precision@K = hits / K              (quality of the top-K list)
      - Recall@K    = hits / |ground_truth|  (coverage of user's liked books)
      - F1@K        = harmonic mean of Precision@K and Recall@K

    Returns: dict with keys 'precision', 'recall', 'f1'
    """
    print(f"\n=== Evaluating KNN (Precision@{top_k}, Recall@{top_k}, F1@{top_k}) ===")
    precisions = []
    recalls    = []
    f1s        = []

    # Sample a subset of active users to speed up evaluation
    active_users = ratings_df[ratings_df['rating'] >= 4]['user_id'].value_counts()
    sample_users = active_users[active_users >= 5].index[:sample_size]

    for user_id in sample_users:
        # Get books this user liked (rating >= 4)
        liked = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]

        if len(liked) < 2:
            continue

        # Use first liked book as "seed" and rest as ground truth
        seed_book_ids = liked['book_id'].values

        seed_id  = seed_book_ids[0]
        seed_row = books_df[books_df['book_id'] == seed_id]
        if seed_row.empty:
            continue
        seed_title = seed_row.iloc[0]['title']

        # Get KNN recommendations
        recs = knn_model.get_recommendations(seed_title, top_k=top_k)
        if recs is None or recs.empty:
            continue

        # Ground truth: the rest of books the user liked (excluding seed)
        ground_truth_titles = set(
            books_df[books_df['book_id'].isin(seed_book_ids[1:])]['title'].values
        )
        rec_titles = set(recs['title'].values[:top_k])

        hits = len(rec_titles & ground_truth_titles)

        # Precision@K = hits / K
        precision = hits / top_k
        precisions.append(precision)

        # Recall@K = hits / |ground_truth|
        n_relevant = len(ground_truth_titles)
        recall = hits / n_relevant if n_relevant > 0 else 0.0
        recalls.append(recall)

        # F1@K = harmonic mean of Precision@K and Recall@K
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall    = np.mean(recalls)    if recalls    else 0
    avg_f1        = np.mean(f1s)        if f1s        else 0

    print(f"  Precision@{top_k} : {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"  Recall@{top_k}    : {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"  F1@{top_k}        : {avg_f1:.4f} ({avg_f1*100:.2f}%)")

    return {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}


def evaluate_svd(svd_model, ratings_df, test_size=0.2):
    """
    Evaluate SVD using RMSE (Root Mean Squared Error).
    Logic: Split ratings into train/test, train SVD on train set, 
           predict on test set, compare to actual ratings.
    WHY RMSE: Standard metric for rating prediction accuracy.
    """
    print("\n=== Evaluating SVD (RMSE) ===")
    
    # Sample a subset of users to speed up evaluation
    sample_users = ratings_df['user_id'].value_counts()[lambda x: x >= 10].index[:200]
    sample_ratings = ratings_df[ratings_df['user_id'].isin(sample_users)]
    
    train_df, test_df = train_test_split(sample_ratings, test_size=test_size, random_state=42)
    
    errors = []
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = row['book_id']
        actual  = row['rating']

        if user_id not in svd_model.user_index or book_id not in svd_model.book_index:
            continue

        u_idx = svd_model.user_index[user_id]
        b_idx = svd_model.book_index[book_id]
        predicted = svd_model.predicted_matrix[u_idx, b_idx]
        predicted = np.clip(predicted, 1, 5)
        errors.append((actual - predicted) ** 2)

    rmse = np.sqrt(np.mean(errors)) if errors else 0
    print(f"SVD RMSE: {rmse:.4f} (lower = better; ideal < 1.0)")
    return rmse


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.preprocessing import load_and_clean_data, create_feature_matrix
    from src.knn_model import KNNRecommender
    from src.svd_model import SVDRecommender

    books, ratings, tags = load_and_clean_data(
        os.path.join('data', 'books.csv'),
        os.path.join('data', 'ratings.csv'),
        os.path.join('data', 'tags.csv')
    )

    # Train models
    feature_matrix, tfidf, scaler = create_feature_matrix(books)
    knn = KNNRecommender(); knn.fit(books, feature_matrix)
    svd = SVDRecommender(); svd.fit(ratings, books)

    # Evaluate
    knn_metrics = evaluate_knn(knn, books, ratings, top_k=10, sample_size=30)
    svd_rmse = evaluate_svd(svd, ratings)

    print("\n=== Final Evaluation Summary ===")
    print(f"  KNN Precision@10 : {knn_metrics['precision']*100:.2f}%")
    print(f"  KNN Recall@10    : {knn_metrics['recall']*100:.2f}%")
    print(f"  KNN F1@10        : {knn_metrics['f1']*100:.2f}%")
    print(f"  SVD RMSE         : {svd_rmse:.4f}")
