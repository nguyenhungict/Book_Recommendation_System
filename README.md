# Book Recommendation System

A Hybrid Recommender System combining **Content-Based Filtering (KNN)** and **Collaborative Filtering (TruncatedSVD)**, deployed as a Flask web application. Built on the GoodBooks-10K dataset (10,000 books, 5,976,479 ratings).

---

## System Architecture

The system consists of four core components:

| Component | Description |
|---|---|
| **Content-Based KNN** | Represents books as 5,003-dim TF-IDF + numerical vectors; uses cosine similarity to find similar books. Works for any user (Cold Start safe). |
| **TruncatedSVD** | Decomposes the sparse user-item rating matrix into 50 latent factors for personalised ranking. Activated for users with ≥ 20 ratings. |
| **Hybrid Router** | Blends both signals: `0.6 × SVD + 0.4 × KNN` for active users; falls back to KNN-only for new/light users. |
| **Smart Intent Detection** | Routes queries through 3 modes: Title Search → Genre Filter → Semantic TF-IDF Discovery. |

---

## Dataset

**GoodBooks-10K Extended** — [Source](https://github.com/malcolmosh/goodbooks-10k-extended)

| File | Description |
|---|---|
| `data/books.csv` | Metadata for 10,000 books (title, authors, genres, description, ratings) |
| `data/ratings.csv` | 5,976,479 explicit ratings (1–5 stars) from 53,424 users |

User-item matrix sparsity: **~98.88%**

---

## Feature Engineering

Each book is represented as a **5,003-dimensional vector**:

- **5,000 dims** — TF-IDF on text soup: `title + 2×authors + 2×genres + description`
- **3 dims** — MinMax-scaled numerical features: `average_rating`, `ratings_count`, `pages`

Sparse matrix stored in CSR format for memory efficiency.

---

## Evaluation Results

### Content-Based KNN — Leave-One-Out Protocol (60 users)

| K | Precision@K | vs. Random Baseline (0.01%) |
|---|---|---|
| 1 | 15.0% | ×1500 |
| 3 | 15.0% | ×1500 |
| 5 | 14.0% | ×1400 |
| **10** | **13.0%** | **×1300** |
| 15 | 8.7% | ×870 |
| 20 | 6.5% | ×650 |

> K=10 chosen as the optimal operating point.

### TruncatedSVD — Train/Test Split (200 users, 80/20)

| Metric | Value |
|---|---|
| RMSE | 2.5626 |
| MAE | 2.2801 |

> Note: High RMSE is expected due to zero-fill bias from sparse matrix factorisation. SVD is used exclusively as a **ranking signal**, not for absolute rating prediction.

---

## Project Structure

```
Book-Recommendation/
├── app.py                    # Flask web application entry point
├── main.py                   # CLI demo script
├── requirements.txt          # Python dependencies
├── regenerate_charts.py      # Script to regenerate all report charts
├── evaluate_and_visualize.py # Evaluation and visualisation utilities
├── test_all.py               # End-to-end test suite
├── report2.tex               # LaTeX final report
│
├── src/
│   ├── preprocessing.py      # Data loading, cleaning, feature engineering
│   ├── knn_model.py          # Content-Based KNN implementation
│   ├── svd_model.py          # TruncatedSVD implementation
│   ├── hybrid.py             # Hybrid Router and Intent Detection
│   └── evaluate.py           # Precision@K and RMSE evaluation
│
├── data/
│   ├── books.csv             # Book metadata
│   ├── ratings.csv           # User ratings
│   ├── book_tags.csv         # Tag assignments
│   └── tags.csv              # Tag definitions
│
└── charts/                   # Generated report figures
    ├── 1_dataset_overview.png
    ├── 2_knn_precision_at_k.png
    ├── 3_svd_rmse_analysis.png
    └── 4_sample_recommendations.png
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/nguyenhungict/Book_Recommendation_System.git
cd Book_Recommendation_System
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web application
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`.

---

## Usage

### Web Application
- **Search by title**: Type a book title to get similar book recommendations.
- **Search by genre**: Type a genre (e.g., `fantasy`, `mystery`) to browse top-rated books in that genre.
- **Free-text discovery**: Describe what you want (e.g., `adventure on a distant planet`) for semantic search.

### CLI Demo
```bash
python main.py
```

### Regenerate Charts
```bash
python regenerate_charts.py
```

---

## Limitations

- **Weighting Imbalance**: The static 60/40 SVD/KNN split can over-prioritise historical user preferences, reducing sensitivity to the current query context for highly active users.
- **SVD Bias**: Zero-filling unobserved entries introduces systematic downward bias in predicted ratings.
- **KNN Scalability**: Brute-force cosine search is O(n×d) per query; FAISS/HNSW indexing would improve latency at scale.
- **No Implicit Feedback**: Only explicit star ratings are used; incorporating click/browse data would enrich both components.

---

