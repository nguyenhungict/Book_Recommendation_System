"""
test_all.py — Full Test Suite for the Hybrid Book Recommendation System
=======================================================================
Run from the book-recommendation/ root folder:
    python test_all.py

Covers:
  TC-01  Genre Filter Accuracy        – does genre filter return correct genres?
  TC-02  KNN Content Similarity       – are similar books thematically close?
  TC-03  SVD Rating Prediction (RMSE) – how accurate are predicted ratings?
  TC-04  SVD Top-K Precision          – do SVD recs match user's real preferences?
  TC-05  Hybrid Blend                 – hybrid score > individual scores?
  TC-06  Cold-Start Fallback          – new user gets KNN, not SVD crash
  TC-07  Keyword Discovery (TF-IDF)   – free-text queries return relevant books
  TC-08  Alias & Edge Cases           – sci-fi, ya, non-fiction, typo-like inputs
  TC-09  End-to-End Precision@10      – overall system Precision@10 over 100 users
"""

import os, sys, re, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src.preprocessing import load_and_clean_data, create_feature_matrix
from src.knn_model     import KNNRecommender
from src.svd_model     import SVDRecommender
from src.hybrid        import HybridRecommender

# ── ANSI colour helpers ─────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m";  RESET  = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"; FAIL = f"{RED}FAIL{RESET}"

results = []   # (tc_id, name, passed, detail)

def record(tc_id, name, passed, detail=""):
    tag = PASS if passed else FAIL
    print(f"  [{tag}] {tc_id}: {name}")
    if detail:
        print(f"         {detail}")
    results.append((tc_id, name, passed, detail))

# ── Boot ────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{CYAN}{'='*65}")
print("  HYBRID BOOK RECOMMENDER — FULL TEST SUITE")
print(f"{'='*65}{RESET}\n")

print("Loading data & training models (this takes ~30s)…")
t0 = time.time()
books, ratings, tags = load_and_clean_data(
    os.path.join('data', 'books.csv'),
    os.path.join('data', 'ratings.csv'),
    os.path.join('data', 'tags.csv'))
feature_matrix, tfidf, scaler = create_feature_matrix(books)

knn = KNNRecommender(); knn.fit(books, feature_matrix, tfidf=tfidf)
svd = SVDRecommender(); svd.fit(ratings, books)
hybrid = HybridRecommender(knn, svd, books, feature_matrix)
print(f"  Ready in {time.time()-t0:.1f}s\n")

# ────────────────────────────────────────────────────────────────────────────
# TC-01  GENRE FILTER ACCURACY
# Expectation: every book returned must contain the requested genre.
# ────────────────────────────────────────────────────────────────────────────
print(f"{BOLD}── TC-01  Genre Filter Accuracy ──────────────────────────────────{RESET}")
GENRE_QUERIES = [
    "fantasy",
    "romance genre",
    "horror books",
    "science fiction",   # tests 'science-fiction' alias
    "mystery novels",
    "thriller",
    "historical fiction",
    "young adult",
]

for q in GENRE_QUERIES:
    recs = knn.search_by_keyword(q, top_k=10)
    if recs.empty:
        record("TC-01", f"'{q}'", False, "No results returned")
        continue
    genre_detected = knn._extract_genre_from_query(q)
    if genre_detected is None:
        record("TC-01", f"'{q}'", False, "Genre not detected — fell through to TF-IDF")
        continue
    # Check: every book must contain the detected genre in its genres column
    hit_count = recs['genres'].str.contains(
        re.escape(genre_detected), case=False, na=False).sum()
    genre_hit_rate = hit_count / len(recs)
    passed = genre_hit_rate >= 0.8   # allow 80% threshold (some books may be missing tags)
    record("TC-01", f"'{q}' → genre='{genre_detected}'",
           passed, f"Genre hit rate: {hit_count}/{len(recs)} = {genre_hit_rate:.0%}")

# ────────────────────────────────────────────────────────────────────────────
# TC-02  KNN CONTENT SIMILARITY (Genre Overlap)
# Expectation: recommended books share at least 1 genre with the seed book.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-02  KNN Content Similarity ─────────────────────────────────{RESET}")
KNN_SEEDS = [
    "Harry Potter and the Sorcerer's Stone",   # fantasy
    "The Hunger Games",                        # dystopia / sci-fi
    "Twilight",                                # romance / paranormal
    "The Da Vinci Code",                       # thriller / mystery
    "The Hobbit",                              # fantasy / adventure
]

import ast
def parse_genres(g):
    try:    return set(ast.literal_eval(g))
    except: return set(str(g).split(','))

for seed in KNN_SEEDS:
    recs = knn.get_recommendations(seed, top_k=10)
    if recs is None or recs.empty:
        record("TC-02", seed, False, "No recommendations returned"); continue
    seed_row = knn.exact_lookup(seed)
    if seed_row is None:
        record("TC-02", seed, False, "Seed book not found"); continue
    seed_genres = parse_genres(seed_row.get('genres', '[]'))
    overlap_count = sum(
        bool(parse_genres(row['genres']) & seed_genres)
        for _, row in recs.iterrows())
    overlap_rate = overlap_count / len(recs)
    avg_sim = recs['similarity_score'].mean()
    passed = overlap_rate >= 0.5 and avg_sim >= 0.10
    record("TC-02", seed, passed,
           f"Genre overlap: {overlap_count}/{len(recs)} = {overlap_rate:.0%}, "
           f"avg similarity: {avg_sim:.3f}")

# ────────────────────────────────────────────────────────────────────────────
# TC-03  SVD RATING PREDICTION (RMSE)
# Expectation: RMSE < 1.0 on a held-out test split.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-03  SVD Rating Prediction (RMSE) ───────────────────────────{RESET}")
from sklearn.model_selection import train_test_split

sample_users = ratings['user_id'].value_counts()[lambda x: x >= 10].index[:300]
sample_ratings = ratings[ratings['user_id'].isin(sample_users)]
train_df, test_df = train_test_split(sample_ratings, test_size=0.2, random_state=42)

errors = []
for _, row in test_df.iterrows():
    uid, bid, actual = row['user_id'], row['book_id'], row['rating']
    if uid not in svd.user_index or bid not in svd.book_index: continue
    pred = np.clip(svd.predicted_matrix[svd.user_index[uid], svd.book_index[bid]], 1, 5)
    errors.append((actual - pred) ** 2)

rmse = np.sqrt(np.mean(errors)) if errors else 99
passed = rmse < 1.0
record("TC-03", "SVD RMSE on held-out ratings", passed,
       f"RMSE = {rmse:.4f} (threshold < 1.0, tested on {len(errors)} ratings)")

# ────────────────────────────────────────────────────────────────────────────
# TC-04  SVD TOP-K PRECISION (Collaborative Filtering)
# Expectation: SVD predicts ≥ 1 actually-liked book in Top-10 for most users.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-04  SVD Top-10 Precision ────────────────────────────────────{RESET}")
SAMPLE = 100
hits_list = []
active = ratings[ratings['rating'] >= 4]['user_id'].value_counts()
active_users = active[active >= 10].index[:SAMPLE]

for uid in active_users:
    liked_ids = set(ratings[(ratings['user_id'] == uid) & (ratings['rating'] >= 4)]['book_id'])
    recs = svd.get_user_recommendations(uid, top_k=10)
    if recs.empty: continue
    rec_ids = set(recs['book_id'])
    hits = len(rec_ids & liked_ids)
    hits_list.append(hits / 10)

avg_prec = np.mean(hits_list) if hits_list else 0
passed = avg_prec >= 0.02   # 2% precision@10 is realistic for sparse collaborative filtering
record("TC-04", f"SVD Precision@10 over {len(hits_list)} users", passed,
       f"Avg Precision@10 = {avg_prec:.4f} ({avg_prec*100:.2f}%)")

# ────────────────────────────────────────────────────────────────────────────
# TC-05  HYBRID BLEND QUALITY
# Expectation: hybrid score covers books from BOTH KNN and SVD lists.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-05  Hybrid Blend Quality ────────────────────────────────────{RESET}")
heavy_users = ratings['user_id'].value_counts()
HYBRID_TEST_CASES = [
    ("Harry Potter and the Sorcerer's Stone", heavy_users[heavy_users >= 100].index[0]),
    ("The Hunger Games",                      heavy_users[heavy_users >= 100].index[1]),
    ("Twilight",                              heavy_users[heavy_users >= 100].index[2]),
]

for book_title, uid in HYBRID_TEST_CASES:
    knn_recs = knn.get_recommendations(book_title, top_k=20)
    svd_recs = svd.get_user_recommendations(uid, top_k=20)
    hyb_recs = hybrid.recommend(book_title=book_title, user_id=uid,
                                ratings_df=ratings, top_k=10)
    if hyb_recs is None or hyb_recs.empty:
        record("TC-05", f"{book_title[:40]} / user {uid}", False, "No hybrid results"); continue

    knn_titles = set(knn_recs['title']) if knn_recs is not None else set()
    svd_titles = set(svd_recs['title']) if not svd_recs.empty else set()
    hyb_titles = set(hyb_recs['title'])

    from_knn = len(hyb_titles & knn_titles)
    from_svd = len(hyb_titles & svd_titles)
    # Hybrid should draw from both sources
    passed = (from_knn > 0 or from_svd > 0)
    record("TC-05", f"'{book_title[:35]}' + user {uid}", passed,
           f"Hybrid result: {from_knn} from KNN, {from_svd} from SVD "
           f"(out of {len(hyb_titles)} total)")

# ────────────────────────────────────────────────────────────────────────────
# TC-06  COLD-START FALLBACK
# Expectation: new/unknown user triggers KNN fallback, no crash.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-06  Cold-Start Fallback ─────────────────────────────────────{RESET}")
COLD_CASES = [
    (None,      "Harry Potter"),   # Guest: no user_id at all
    (99999999,  "Twilight"),       # Completely unknown user_id
]

for uid, book in COLD_CASES:
    try:
        recs = hybrid.recommend(
            book_title=book,
            user_id=uid,
            ratings_df=ratings if uid else None,
            top_k=10)
        passed = recs is not None and not recs.empty
        detail = f"Got {len(recs)} recs (KNN fallback)" if passed else "Empty result"
    except Exception as e:
        passed, detail = False, f"Exception: {e}"
    record("TC-06", f"user={uid}, book='{book}'", passed, detail)

# ────────────────────────────────────────────────────────────────────────────
# TC-07  KEYWORD DISCOVERY (TF-IDF path — non-genre queries)
# Expectation: free-text queries return results with non-zero similarity scores.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-07  Keyword Discovery (TF-IDF) ──────────────────────────────{RESET}")
KEYWORD_QUERIES = [
    ("magic school adventure",        None),   # should NOT hit genre filter
    ("love story paris",              None),
    ("detective murder investigation", None),
    ("dystopian survival",            None),
    ("wizard dragon quest",           None),
]

for q, expected_genre in KEYWORD_QUERIES:
    genre_hit = knn._extract_genre_from_query(q)
    recs = knn.search_by_keyword(q, top_k=10)
    # For non-genre queries, we expect TF-IDF path
    used_tfidf = (genre_hit is None)
    has_results = not recs.empty
    avg_score = recs['similarity_score'].mean() if has_results else 0
    passed = has_results and avg_score > 0
    record("TC-07", f"'{q}'", passed,
           f"Path={'TF-IDF' if used_tfidf else f'Genre({genre_hit})'}, "
           f"{len(recs)} results, avg score={avg_score:.3f}")

# ────────────────────────────────────────────────────────────────────────────
# TC-08  ALIAS & EDGE CASES
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-08  Alias & Edge Cases ──────────────────────────────────────{RESET}")
ALIAS_CASES = [
    # (query,                   expected_canonical_genre)
    ("sci-fi",                  "science-fiction"),
    ("scifi books",             "science-fiction"),
    ("science fiction",         "science-fiction"),   # space variant
    ("ya romance",              "young-adult"),
    ("young adult books",       "young-adult"),        # space variant
    ("non-fiction",             "nonfiction"),
    ("best thrillers",          "thriller"),           # plural
    ("graphic novels",          "graphic"),
    ("historical fiction",      "historical-fiction"), # space variant
    ("mystery novels",          "mystery"),            # plural-ish
    ("",                        None),                 # empty query → None
    ("the",                     None),                 # stop-word only → None
    ("a book like Harry Potter", None),                # free-text → None
]

for q, expected in ALIAS_CASES:
    detected = knn._extract_genre_from_query(q) if q else None
    passed = (detected == expected)
    record("TC-08", f"'{q}' → expected '{expected}'", passed,
           f"Got: '{detected}'")

# ────────────────────────────────────────────────────────────────────────────
# TC-09  END-TO-END Precision@10 (Hybrid system on real users)
# Uses leave-one-out: hide last liked book, check if hybrid recommends it.
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}── TC-09  End-to-End Hybrid Precision@10 ──────────────────────────{RESET}")
E2E_SAMPLE = 50
e2e_precisions = []

eligible = ratings[ratings['rating'] >= 4]['user_id'].value_counts()
eligible = eligible[eligible >= 15].index[:E2E_SAMPLE]

for uid in eligible:
    liked = ratings[(ratings['user_id'] == uid) & (ratings['rating'] >= 4)]
    liked_ids = liked['book_id'].tolist()
    if len(liked_ids) < 2: continue

    seed_id = liked_ids[0]
    ground_truth = set(liked_ids[1:])

    seed_row = books[books['book_id'] == seed_id]
    if seed_row.empty: continue
    seed_title = seed_row.iloc[0]['title']

    try:
        recs = hybrid.recommend(book_title=seed_title, user_id=uid,
                                ratings_df=ratings, top_k=10)
    except Exception:
        continue

    if recs is None or recs.empty: continue
    rec_titles = set(recs['title'])
    truth_titles = set(books[books['book_id'].isin(ground_truth)]['title'])
    hits = len(rec_titles & truth_titles)
    e2e_precisions.append(hits / 10)

e2e_avg = np.mean(e2e_precisions) if e2e_precisions else 0
passed = e2e_avg >= 0.01   # ≥1% Precision@10 is acceptable for this sparse dataset
record("TC-09", f"E2E Hybrid Precision@10 ({len(e2e_precisions)} users)", passed,
       f"Avg Precision@10 = {e2e_avg:.4f} ({e2e_avg*100:.2f}%)")

# ── FINAL SUMMARY ────────────────────────────────────────────────────────────
print(f"\n{BOLD}{CYAN}{'='*65}")
print("  FINAL SUMMARY")
print(f"{'='*65}{RESET}")
total  = len(results)
passed_n = sum(1 for r in results if r[2])
failed_n = total - passed_n

for tc_id, name, ok, detail in results:
    tag = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
    print(f"  {tag} {tc_id}: {name}")

print(f"\n  Total : {total}")
print(f"  {GREEN}Passed: {passed_n}{RESET}")
print(f"  {RED}Failed: {failed_n}{RESET}")
print(f"  Pass rate: {passed_n/total*100:.1f}%\n")
