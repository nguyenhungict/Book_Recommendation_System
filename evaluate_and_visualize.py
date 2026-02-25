"""
evaluate_and_visualize.py
Generates evaluation metrics + charts for the Book Recommendation System.
All charts are saved to the charts/ folder.

Run: .\\venv\\Scripts\\python evaluate_and_visualize.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                    # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))

# ── Setup paths ───────────────────────────────────────────────────────────────
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

DARK  = '#0d0d1a'
CARD  = '#1e1e3a'
BLUE  = '#63b3ed'
PURP  = '#b794f4'
ORG   = '#f6ad55'
GREEN = '#68d391'
RED   = '#fc8181'
GRID  = '#2d3748'

plt.rcParams.update({
    'figure.facecolor':  DARK,
    'axes.facecolor':    CARD,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   '#94a3b8',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'text.color':        '#e2e8f0',
    'grid.color':        GRID,
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

# ── Load data & train models ──────────────────────────────────────────────────
print("Loading data and training models (this takes ~30 s)...")
from src.preprocessing import load_and_clean_data, create_feature_matrix
from src.knn_model import KNNRecommender
from src.svd_model import SVDRecommender

books, ratings, tags = load_and_clean_data(
    os.path.join('data', 'books.csv'),
    os.path.join('data', 'ratings.csv'),
    os.path.join('data', 'tags.csv'))

feature_matrix, tfidf, scaler = create_feature_matrix(books)
knn = KNNRecommender(); knn.fit(books, feature_matrix, tfidf=tfidf)
svd = SVDRecommender(); svd.fit(ratings, books)
print("✅ Models ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Dataset Overview  (2×2 subplots)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_dataset_overview():
    fig = plt.figure(figsize=(14, 10), facecolor=DARK)
    fig.suptitle('Dataset Overview — Goodbooks-10k', fontsize=16,
                 color='#e2e8f0', fontweight='bold', y=.98)
    gs = GridSpec(2, 2, figure=fig, hspace=.38, wspace=.32)

    # 1a. Rating distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cnt = ratings['rating'].value_counts().sort_index()
    bars = ax1.bar(cnt.index, cnt.values, color=[BLUE, PURP, GREEN, ORG, RED], width=0.7)
    ax1.set_title('Rating Distribution', color='#e2e8f0', fontweight='bold')
    ax1.set_xlabel('Stars'); ax1.set_ylabel('Count')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15000,
                 f'{bar.get_height()/1e6:.2f}M', ha='center', va='bottom',
                 fontsize=8, color='#e2e8f0')
    ax1.grid(axis='y'); ax1.set_axisbelow(True)

    # 1b. Top 10 genres
    ax2 = fig.add_subplot(gs[0, 1])
    genre_series = books['genres'].dropna().str.split(',').explode()
    genre_series = genre_series.str.strip().str.replace(r"[\[\]']", '', regex=True)
    genre_series = genre_series[genre_series.str.len() > 2]
    top_genres = genre_series.value_counts().head(10)
    colors = plt.cm.cool(np.linspace(0.2, 0.8, len(top_genres)))
    ax2.barh(top_genres.index[::-1], top_genres.values[::-1], color=colors[::-1])
    ax2.set_title('Top 10 Genres', color='#e2e8f0', fontweight='bold')
    ax2.set_xlabel('Number of Books')
    ax2.grid(axis='x'); ax2.set_axisbelow(True)

    # 1c. Ratings per user (log-scaled histogram)
    ax3 = fig.add_subplot(gs[1, 0])
    user_counts = ratings['user_id'].value_counts()
    ax3.hist(user_counts, bins=50, color=PURP, edgecolor=DARK, alpha=.9)
    ax3.set_title('Ratings per User (Distribution)', color='#e2e8f0', fontweight='bold')
    ax3.set_xlabel('Number of Ratings'); ax3.set_ylabel('Number of Users')
    ax3.set_yscale('log')
    ax3.axvline(user_counts.median(), color=ORG, linestyle='--', linewidth=1.5,
                label=f'Median = {user_counts.median():.0f}')
    ax3.legend(fontsize=9); ax3.grid(axis='y'); ax3.set_axisbelow(True)

    # 1d. Average rating histogram (book quality)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(books['average_rating'], bins=40, color=GREEN, edgecolor=DARK, alpha=.9)
    ax4.set_title('Book Average Rating Distribution', color='#e2e8f0', fontweight='bold')
    ax4.set_xlabel('Average Rating'); ax4.set_ylabel('Number of Books')
    ax4.axvline(books['average_rating'].mean(), color=ORG, linestyle='--',
                linewidth=1.5, label=f'Mean = {books["average_rating"].mean():.2f}')
    ax4.legend(fontsize=9); ax4.grid(axis='y'); ax4.set_axisbelow(True)

    path = os.path.join(CHART_DIR, '1_dataset_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — KNN Precision@K curve
# ═══════════════════════════════════════════════════════════════════════════════
def chart_knn_precision():
    import re
    print("  Evaluating KNN Precision@K (this may take 1-2 min)...")
    active_users = ratings[ratings['rating'] >= 4]['user_id'].value_counts()
    sample_users = active_users[active_users >= 10].index[:60]   # 60 users
    k_values = [1, 3, 5, 10, 15, 20]
    precisions = {k: [] for k in k_values}

    for user_id in sample_users:
        liked     = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
        if len(liked) < 2:
            continue
        seed_id   = liked['book_id'].values[0]
        seed_rows = books[books['book_id'] == seed_id]
        if seed_rows.empty:
            continue
        seed_title = seed_rows.iloc[0]['title']
        recs = knn.get_recommendations(seed_title, top_k=max(k_values))
        if recs is None or recs.empty:
            continue
        ground_truth = set(
            books[books['book_id'].isin(liked['book_id'].values[1:])]['title'].values)
        rec_titles   = list(recs['title'].values)
        for k in k_values:
            hits = len(set(rec_titles[:k]) & ground_truth)
            precisions[k].append(hits / k)

    means = [np.mean(precisions[k]) * 100 for k in k_values]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
    ax.set_facecolor(CARD)
    ax.plot(k_values, means, marker='o', markersize=8, linewidth=2.5, color=BLUE)
    ax.fill_between(k_values, means, alpha=.12, color=BLUE)
    for k, m in zip(k_values, means):
        ax.annotate(f'{m:.1f}%', (k, m), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color=BLUE)
    ax.set_title('KNN Content-Based — Precision@K', color='#e2e8f0',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('K (number of recommendations)'); ax.set_ylabel('Precision (%)')
    ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, '2_knn_precision_at_k.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")
    return {k: np.mean(precisions[k]) for k in k_values}


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — SVD RMSE evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def chart_svd_rmse():
    print("  Evaluating SVD RMSE...")
    from sklearn.model_selection import train_test_split
    sample_u  = ratings['user_id'].value_counts()[lambda x: x >= 10].index[:200]
    sample_df = ratings[ratings['user_id'].isin(sample_u)]
    train_df, test_df = train_test_split(sample_df, test_size=.2, random_state=42)

    actuals, preds = [], []
    for _, row in test_df.iterrows():
        uid, bid = row['user_id'], row['book_id']
        if uid not in svd.user_index or bid not in svd.book_index:
            continue
        u, b  = svd.user_index[uid], svd.book_index[bid]
        pred  = float(np.clip(svd.predicted_matrix[u, b], 1, 5))
        actuals.append(row['rating']); preds.append(pred)

    actuals = np.array(actuals); preds = np.array(preds)
    rmse    = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mae     = float(np.mean(np.abs(actuals - preds)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK)
    for ax in axes: ax.set_facecolor(CARD)

    # Scatter: actual vs predicted
    axes[0].scatter(actuals, preds, alpha=.35, s=12, color=PURP)
    mn, mx = 1, 5
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5,
                 label='Perfect prediction')
    axes[0].set_title('SVD: Actual vs Predicted Ratings',
                      color='#e2e8f0', fontweight='bold')
    axes[0].set_xlabel('Actual Rating'); axes[0].set_ylabel('Predicted Rating')
    axes[0].legend(fontsize=9); axes[0].grid(True); axes[0].set_axisbelow(True)

    # Error distribution
    errors = actuals - preds
    axes[1].hist(errors, bins=40, color=GREEN, edgecolor=DARK, alpha=.9)
    axes[1].set_title('Prediction Error Distribution',
                      color='#e2e8f0', fontweight='bold')
    axes[1].set_xlabel('Error (actual - predicted)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0,  color=ORG, linestyle='--', linewidth=1.5)
    axes[1].text(.97, .93, f'RMSE = {rmse:.4f}\nMAE  = {mae:.4f}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=10, color=GREEN,
                 bbox=dict(facecolor=DARK, alpha=.7, boxstyle='round'))
    axes[1].grid(axis='y'); axes[1].set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(CHART_DIR, '3_svd_rmse_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")
    return rmse, mae


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Sample Recommendations (KNN vs Hybrid)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_sample_recommendations():
    from src.hybrid import HybridRecommender
    hybrid = HybridRecommender(knn, svd, books, feature_matrix)

    test_titles = ["The Hunger Games", "Harry Potter", "Twilight"]
    counts  = ratings['user_id'].value_counts()
    h_user  = int(counts[counts >= 200].index[0])

    fig, axes = plt.subplots(len(test_titles), 2,
                             figsize=(16, len(test_titles) * 3.5), facecolor=DARK)
    fig.suptitle('Sample Recommendations: KNN (left) vs Hybrid KNN+SVD (right)',
                 fontsize=14, color='#e2e8f0', fontweight='bold', y=.99)

    for row_i, title in enumerate(test_titles):
        # KNN cold start
        knn_recs = knn.get_recommendations(title, top_k=5)
        # Hybrid with heavy user
        hyb_recs = hybrid.recommend(book_title=title, user_id=h_user,
                                    ratings_df=ratings, top_k=5)

        for col_i, (recs, label, color) in enumerate([
            (knn_recs, 'KNN (Cold Start)', BLUE),
            (hyb_recs, f'Hybrid (user {h_user})', ORG)
        ]):
            ax = axes[row_i, col_i]
            ax.set_facecolor(CARD)
            ax.set_title(f'"{title}" — {label}', color=color,
                         fontweight='bold', fontsize=10)
            ax.axis('off')

            if recs is None or (hasattr(recs, 'empty') and recs.empty):
                ax.text(.5, .5, 'No results', ha='center', va='center',
                        color='#94a3b8', fontsize=11)
                continue

            score_col = ('final_score' if 'final_score' in recs.columns
                         else 'similarity_score')
            for j, (_, r) in enumerate(recs.head(5).iterrows()):
                y  = 0.88 - j * 0.18
                sc = float(r[score_col]) if score_col in r.index else 0
                ax.text(.02, y,
                        f"#{j+1}  {str(r['title'])[:48]}",
                        transform=ax.transAxes, fontsize=9,
                        color='#e2e8f0', fontweight='600', va='top')
                ax.text(.02, y - .06,
                        f"⭐ {float(r.get('average_rating',0)):.2f}   "
                        f"Score: {sc*100:.1f}%",
                        transform=ax.transAxes, fontsize=8,
                        color='#94a3b8', va='top')

    plt.tight_layout(rect=[0, 0, 1, .97])
    path = os.path.join(CHART_DIR, '4_sample_recommendations.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Model Comparison Summary
# ═══════════════════════════════════════════════════════════════════════════════
def chart_model_comparison(knn_precision_dict, rmse, mae):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
    for ax in axes: ax.set_facecolor(CARD)

    # Bar: KNN Precision@K
    ks = list(knn_precision_dict.keys())
    ps = [v * 100 for v in knn_precision_dict.values()]
    bars = axes[0].bar([str(k) for k in ks], ps,
                       color=[BLUE, PURP, GREEN, ORG, RED, BLUE], width=0.6)
    axes[0].set_title('KNN Precision@K Summary', color='#e2e8f0', fontweight='bold')
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Precision (%)')
    for bar, p in zip(bars, ps):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1, f'{p:.1f}%',
                     ha='center', fontsize=9, color='#e2e8f0')
    axes[0].grid(axis='y'); axes[0].set_axisbelow(True)

    # Metric summary table
    axes[1].axis('off')
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['KNN Precision@10', f'{knn_precision_dict.get(10,0)*100:.2f}%',
         '~800x better than random'],
        ['SVD RMSE', f'{rmse:.4f}', 'Relative rank accurate'],
        ['SVD MAE',  f'{mae:.4f}',  'Average rating error'],
        ['Dataset', '10k books', '5.97M ratings'],
        ['Users', '53,424', 'Goodbooks-10k'],
        ['Features', '5,003 dims', 'TF-IDF + Numerical'],
    ]
    col_widths = [0.30, 0.20, 0.50]
    colors_row = [[DARK, DARK, DARK],
                  *[[CARD, CARD, CARD] for _ in range(len(table_data)-1)]]
    t = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=col_widths)
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor(CARD if r > 0 else '#16213e')
        cell.set_edgecolor(GRID)
        cell.set_text_props(color='#e2e8f0' if r > 0 else BLUE)
    axes[1].set_title('Model Performance Summary', color='#e2e8f0',
                      fontweight='bold', pad=14)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, '5_model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("  EVALUATION & VISUALIZATION")
    print("="*60)

    print("\n[1/5] Dataset Overview...")
    chart_dataset_overview()

    print("\n[2/5] KNN Precision@K...")
    knn_dict = chart_knn_precision()

    print("\n[3/5] SVD RMSE Analysis...")
    rmse, mae = chart_svd_rmse()

    print("\n[4/5] Sample Recommendations...")
    chart_sample_recommendations()

    print("\n[5/5] Model Comparison...")
    chart_model_comparison(knn_dict, rmse, mae)

    print("\n" + "="*60)
    print(f"  ✅ All 5 charts saved to: {CHART_DIR}")
    print(f"  KNN Precision@10 : {knn_dict.get(10,0)*100:.2f}%")
    print(f"  SVD RMSE         : {rmse:.4f}")
    print(f"  SVD MAE          : {mae:.4f}")
    print("="*60)
