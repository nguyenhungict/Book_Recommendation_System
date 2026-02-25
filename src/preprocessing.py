import pandas as pd
import numpy as np

def load_and_clean_data(books_path, ratings_path, tags_path):
    """
    Load data from CSVs and perform basic cleaning: Missing values and formatting.
    WHY: Handles raw files before they are fed into TF-IDF or SVD models.
    """
    print("Loading datasets...")
    # Load dataset
    books_df = pd.read_csv(books_path)
    ratings_df = pd.read_csv(ratings_path)
    tags_df = pd.read_csv(tags_path)
    
    # 1. Handle missing values in books.csv
    # Fill missing text features with empth string so TF-IDF doesn't crash 
    text_columns = ['authors', 'title', 'description', 'genres']
    for col in text_columns:
        if col in books_df.columns:
            books_df[col] = books_df[col].fillna('')
            
    # Fill missing numerical features with mean or median
    if 'average_rating' in books_df.columns:
        books_df['average_rating'] = books_df['average_rating'].fillna(books_df['average_rating'].mean())
    if 'pages' in books_df.columns:
        books_df['pages'] = books_df['pages'].fillna(books_df['pages'].median())
    if 'language_code' in books_df.columns:
        books_df['language_code'] = books_df['language_code'].fillna('eng') # Default language
        
    print(f"Data Loaded: {len(books_df)} books, {len(ratings_df)} ratings, {len(tags_df)} tags.")
    return books_df, ratings_df, tags_df

def create_feature_matrix(books_df):
    """
    Combines Text features (TF-IDF on description, title, authors, genres) 
    and Numerical Features (Rating, Pages count) into a single Vector Matrix.
    WHY: KNN needs all features to be numbers on a similar scale to calculate distance.
    """
    print("Starting Feature Engineering...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MinMaxScaler
    from scipy.sparse import hstack

    # 1. Combine Text Features
    # We create a 'soup' of text. We give more weight to authors and genres by repeating them.
    books_df['text_soup'] = (
        books_df['title'] + " " + 
        books_df['authors'] + " " + books_df['authors'] + " " + 
        books_df['genres'] + " " + books_df['genres'] + " " + 
        books_df['description']
    )
    
    # 2. Text Vectorization (TF-IDF)
    print("- Vectorizing Text Features (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books_df['text_soup'])
    
    # 3. Numerical Feature Scaling (MinMaxScaler)
    print("- Scaling Numerical Features...")
    scaler = MinMaxScaler()
    num_features = books_df[['average_rating', 'pages', 'ratings_count']].fillna(0)
    num_matrix = scaler.fit_transform(num_features)
    
    # 4. Combine Both Matrices
    # hstack is used because TF-IDF creates a sparse matrix, which we must maintain to save RAM.
    print("- Combining Features into final matrix...")
    from scipy.sparse import csr_matrix
    num_sparse_matrix = csr_matrix(num_matrix)
    final_feature_matrix = hstack((tfidf_matrix, num_sparse_matrix))
    
    print(f"Feature Engineering Done: Matrix shape = {final_feature_matrix.shape}")
    return final_feature_matrix, tfidf, scaler

if __name__ == "__main__":
    """
    Run this file directly to test data loading and feature engineering.
    Usage: python src/preprocessing.py
          (Must be run from the root  book-recommendation/ folder)
    """
    import os

    # Step 1: Load and clean data
    books, ratings, tags = load_and_clean_data(
        os.path.join('data', 'books.csv'),
        os.path.join('data', 'ratings.csv'),
        os.path.join('data', 'tags.csv')
    )

    # Step 2: Quick overview of the loaded data
    print("\n--- Books Sample ---")
    print(books[['title', 'authors', 'genres', 'average_rating', 'pages']].head(3))
    print(f"\nMissing values in books:\n{books[['description', 'genres', 'pages']].isnull().sum()}")

    print("\n--- Ratings Sample ---")
    print(ratings.head(3))
    print(f"Rating scale: {ratings['rating'].min()} → {ratings['rating'].max()}")

    # Step 3: Build feature matrix
    feature_matrix, tfidf, scaler = create_feature_matrix(books)
    print(f"\nFinal Feature Matrix: {feature_matrix.shape[0]} books × {feature_matrix.shape[1]} features")
    print("Preprocessing complete! Ready for Model training.")
