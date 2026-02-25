import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_knn(knn_model, books_df, ratings_df, top_k=10, sample_size=50):
    """
    Evaluate KNN using Precision@K.
    Logic: For a sample of users, treat their highest-rated books as 'ground truth'.
           Check how many of KNN's suggestions appear in that ground truth.
    WHY Precision@K: Standard metric for recommendation systems.
    """
    print(f"\n=== Evaluating KNN (Precision@{top_k}) ===")
    precisions = []
    
    # Sample a subset of active users to speed up evaluation
    active_users = ratings_df[ratings_df['rating'] >= 4]['user_id'].value_counts()
    sample_users = active_users[active_users >= 5].index[:sample_size]
    
    for user_id in sample_users:
        # Get books this user liked (rating >= 4) 
        liked = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]
        
        if liked.empty:
            continue
            
        # Use first liked book as "seed" and rest as ground truth
        seed_book_ids = liked['book_id'].values
        
        # Get book title for a random seed
        seed_id = seed_book_ids[0]
        seed_row = books_df[books_df['book_id'] == seed_id]
        if seed_row.empty:
            continue
        seed_title = seed_row.iloc[0]['title']
        
        # Get KNN recommendations
        recs = knn_model.get_recommendations(seed_title)
        if recs is None or recs.empty:
            continue
        
        # Ground truth: the rest of books the user liked
        ground_truth_titles = set(
            books_df[books_df['book_id'].isin(seed_book_ids[1:])]['title'].values
        )
        rec_titles = set(recs['title'].values[:top_k])
        
        # Precision = how many recommendations are in ground truth
        hits = len(rec_titles & ground_truth_titles)
        precision = hits / top_k
        precisions.append(precision)
    
    avg_precision = np.mean(precisions) if precisions else 0
    print(f"KNN Precision@{top_k}: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    return avg_precision


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
    knn_precision = evaluate_knn(knn, books, ratings, top_k=10, sample_size=30)
    svd_rmse = evaluate_svd(svd, ratings)

    print("\n=== Final Evaluation Summary ===")
    print(f"  KNN Precision@10 : {knn_precision*100:.2f}%")
    print(f"  SVD RMSE         : {svd_rmse:.4f}")
