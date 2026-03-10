import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import pickle
import os
import re

# ── All known genres in the Goodbooks-10k dataset ──────────────────────────────
# IMPORTANT: This is a LIST sorted by length descending (longest first).
# WHY: When detecting genre intent from a query, we must check longer/more
# specific genres (e.g. 'science-fiction', 'historical-fiction') BEFORE
# shorter/general ones ('science', 'historical', 'fiction').
# Using a set would iterate in random order → wrong genre matched randomly.
_GENRE_ALIASES = {
    # Alias (what user types) → Canonical (what's in genres column)
    'sci-fi':       'science-fiction',
    'scifi':        'science-fiction',
    'science fiction': 'science-fiction',  # space variant
    'ya':           'young-adult',
    'young adult':  'young-adult',         # space variant
    'non-fiction':  'nonfiction',
    'literary':     'fiction',
    'historical fiction': 'historical-fiction',  # space variant
}

_GENRE_RAW = [
    'historical-fiction', 'science-fiction', 'young-adult',
    'chick-lit', 'self-help', 'paranormal', 'spirituality',
    'contemporary', 'philosophy', 'psychology', 'biography',
    'nonfiction', 'suspense', 'christian', 'classics', 'religion',
    'business', 'thriller', 'mystery', 'romance', 'fantasy',
    'history', 'historical', 'graphic', 'comics', 'horror',
    'poetry', 'memoir', 'sports', 'travel', 'humor', 'music',
    'manga', 'crime', 'science', 'fiction', 'art',
]

# Sorted longest → shortest so specific genres always win over general ones.
KNOWN_GENRES = sorted(
    set(_GENRE_RAW) | set(_GENRE_ALIASES.keys()),
    key=len, reverse=True
)


class KNNRecommender:
    """
    Content-Based Recommendation using K-Nearest Neighbors.
    Finds similarity between books based on TF-IDF (text) + Scaled Ratings (numbers).
    """
    def __init__(self, n_neighbors=21):
        # 21 because position [0] is always the query book itself (distance=0)
        # so effective max top_k = 20
        self.n_neighbors    = n_neighbors
        self.model          = NearestNeighbors(n_neighbors=n_neighbors,
                                               metric='cosine', algorithm='brute')
        self.books_df       = None
        self.feature_matrix = None
        self.tfidf          = None   # stored for keyword search

    # ──────────────────────────────────────────────────────────────
    def fit(self, books_df, feature_matrix, tfidf=None):
        """Train the KNN model on the pre-built feature matrix."""
        print("Training KNN Model...")
        self.books_df       = books_df
        self.feature_matrix = feature_matrix
        self.tfidf          = tfidf
        self.model.fit(feature_matrix)
        print("KNN Model Trained Successfully!")

    # ──────────────────────────────────────────────────────────────
    def exact_lookup(self, book_title):
        """
        Returns the first book row that contains 'book_title' (case-insensitive).
        Returns None when no match is found.
        WHY: Used by the UI to detect 'Search mode' vs 'Discovery mode'.
        """
        safe = re.escape(book_title.strip())
        matched = self.books_df[
            self.books_df['title'].str.contains(safe, case=False, na=False)]
        return None if matched.empty else matched.iloc[0]

    # ──────────────────────────────────────────────────────────────
    def get_book_detail(self, book_title):
        """Return metadata dict for one book (for the Book Detail card in UI)."""
        row = self.exact_lookup(book_title)
        if row is None:
            return None
        return {
            'book_id':        int(row.get('book_id', 0)),
            'title':          str(row.get('title', '')),
            'authors':        str(row.get('authors', '')),
            'genres':         str(row.get('genres', '')),
            'average_rating': float(row.get('average_rating', 0)),
            'ratings_count':  int(row.get('ratings_count', 0)),
            'description':    str(row.get('description', '')),
        }

    # ──────────────────────────────────────────────────────────────
    def get_recommendations(self, book_title, top_k=10):
        """
        Given an exact (or near-exact) book title, return top similar books.
        Skips the query book itself in results.
        """
        safe = re.escape(book_title.strip())
        matched = self.books_df[
            self.books_df['title'].str.contains(safe, case=False, na=False)]

        if matched.empty:
            print(f"Book '{book_title}' not found in the dataset.")
            return None

        book_idx     = matched.index[0]
        actual_title = matched.iloc[0]['title']
        print(f"Found match: {actual_title}")

        book_vector = self.feature_matrix[book_idx]
        n = min(self.n_neighbors, self.feature_matrix.shape[0])
        distances, indices = self.model.kneighbors(book_vector, n_neighbors=n)

        # Skip index [0] = the book itself
        rec_idx   = indices[0][1:]
        rec_dist  = distances[0][1:]

        recs = self.books_df.iloc[rec_idx][
            ['title', 'authors', 'genres', 'average_rating']].copy()
        recs['similarity_score'] = 1 - rec_dist
        return recs.head(top_k)

    # ──────────────────────────────────────────────────────────────
    def _extract_genre_from_query(self, query):
        """
        Stage-1 Intent Detection: check if the query contains a known genre name.
        Returns the matched genre string, or None if no genre detected.

        Examples:
          'fantasy genre'         → 'fantasy'
          'best sci-fi books'     → 'science-fiction'  (alias mapping)
          'ya romance'            → 'young-adult' + 'romance'
          'a book like HP'        → None  (fallback to TF-IDF)
        WHY: Genre names are too common in description text; TF-IDF cannot
             distinguish 'the fantasy genre' in a description from an actual
             fantasy book.
        """
        q_lower = query.lower()

        # Step 1: Check aliases first (e.g. 'sci-fi' → 'science-fiction')
        # Aliases are checked before the main list to avoid matching partial words.
        for alias, canonical in _GENRE_ALIASES.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', q_lower):
                return canonical

        # Step 2: Check known genres in length-descending order.
        # WHY ordered: 'science-fiction' must be checked before 'science' and 'fiction',
        # 'historical-fiction' before 'historical' and 'fiction', etc.
        # KNOWN_GENRES is already sorted longest→shortest (see module level).
        # Also try stripping a trailing 's' to handle plurals:
        #   'thrillers' → 'thriller', 'mysteries' → 'mystery' (after strip)
        q_variants = [q_lower]
        if q_lower.endswith('s') and not q_lower.endswith('ss'):
            q_variants.append(q_lower[:-1])   # e.g. 'thrillers' → 'thriller'

        for g in KNOWN_GENRES:
            if g in _GENRE_ALIASES:       # skip alias keys, handled above
                continue
            pattern = r'\b' + re.escape(g) + r'\b'
            for qv in q_variants:
                if re.search(pattern, qv):
                    return g
        return None

    # ──────────────────────────────────────────────────────────────
    def search_by_genre(self, genre, top_k=10):
        """
        Direct genre filter: returns top books whose `genres` column contains
        the requested genre, sorted by average_rating descending.
        WHY: Exact genre matching is more reliable than TF-IDF for genre queries.

        Note: We reorder each book's genres list to put the matched genre FIRST.
        This is important because the UI's trimGenres() only shows the first 2
        genres — without reordering, books where 'romance' is in position 3+
        would appear to have no romance tag at all.
        """
        mask = self.books_df['genres'].str.contains(
            re.escape(genre), case=False, na=False)
        results = self.books_df[mask][
            ['title', 'authors', 'genres', 'average_rating']].copy()
        results = results.sort_values('average_rating', ascending=False).head(top_k)

        # Reorder genres so the matched genre always appears first
        def _promote_genre(genres_str, target):
            """Move `target` to the front of the genres list string."""
            try:
                import ast
                genres_list = ast.literal_eval(genres_str)
                # find the genre (case-insensitive)
                matched = [g for g in genres_list if target.lower() in g.lower()]
                others  = [g for g in genres_list if target.lower() not in g.lower()]
                return str(matched + others)
            except Exception:
                return genres_str

        results['genres'] = results['genres'].apply(
            lambda g: _promote_genre(g, genre))

        results['similarity_score'] = 1.0   # perfect genre match → score = 1.0
        return results.reset_index(drop=True)


    # ──────────────────────────────────────────────────────────────
    def search_by_keyword(self, query, top_k=10):
        """
        Discovery mode with 2-stage routing:
          Stage 1 — Genre Detection : if query contains a known genre,
                    filter directly on the `genres` column (precise).
          Stage 2 — TF-IDF Search  : for all other free-text queries.

        WHY 2-stage: TF-IDF matches the word 'fantasy' or 'genre' anywhere in
        the text_soup (including description), causing false positives when the
        user clearly wants books OF a genre, not books MENTIONING that genre.
        """
        # -- Stage 1: Genre-intent detection --
        detected_genre = self._extract_genre_from_query(query)
        if detected_genre:
            print(f"[Genre Filter] Detected genre intent: '{detected_genre}' — skipping TF-IDF.")
            results = self.search_by_genre(detected_genre, top_k=top_k)
            if not results.empty:
                return results
            print(f"[Genre Filter] No books found for genre '{detected_genre}', falling back to TF-IDF.")

        # -- Stage 2: TF-IDF semantic search (fallback) --
        if self.tfidf is None:
            print("TF-IDF vectorizer not stored. Pass tfidf= when calling fit().")
            return pd.DataFrame()

        print(f"[TF-IDF] Semantic search for: '{query}'")
        # Transform query into TF-IDF space
        query_tfidf = self.tfidf.transform([query])          # sparse (1 × V)

        # Pad zeros for the numerical columns appended during preprocessing
        n_num = self.feature_matrix.shape[1] - query_tfidf.shape[1]
        if n_num > 0:
            pad       = sp.csr_matrix((1, n_num))
            query_vec = sp.hstack([query_tfidf, pad])
        else:
            query_vec = query_tfidf

        sims    = cosine_similarity(query_vec, self.feature_matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]

        results = self.books_df.iloc[top_idx][
            ['title', 'authors', 'genres', 'average_rating']].copy()
        results['similarity_score'] = sims[top_idx]
        return results.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    def save_model(self, path="models/knn_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load_model(self, path="models/knn_model.pkl"):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.preprocessing import load_and_clean_data, create_feature_matrix

    books, ratings, tags = load_and_clean_data(
        os.path.join('data', 'books.csv'),
        os.path.join('data', 'ratings.csv'),
        os.path.join('data', 'tags.csv')
    )
    feature_matrix, tfidf, scaler = create_feature_matrix(books)

    knn = KNNRecommender()
    knn.fit(books, feature_matrix, tfidf=tfidf)

    print("\n--- Exact book recommendations ---")
    print(knn.get_recommendations("Harry Potter").to_string(index=False))

    print("\n--- Keyword search: 'fantasy adventure' ---")
    print(knn.search_by_keyword("fantasy adventure").to_string(index=False))
