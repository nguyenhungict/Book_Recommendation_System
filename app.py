"""
app.py — Smart Book Recommender with Intent Detection
Two modes:
  • SEARCH mode  : User types an exact book title → show Book Detail card + "You might also like" recommendations
  • DISCOVER mode: User types a free-text query   → semantic TF-IDF search → results list
"""
import os, re
from flask import Flask, request, jsonify, render_template_string

from src.preprocessing import load_and_clean_data, create_feature_matrix
from src.knn_model import KNNRecommender
from src.svd_model import SVDRecommender
from src.hybrid import HybridRecommender

app = Flask(__name__)

# ── Boot: load data + train models once ────────────────────────────────────────
print("Loading data...")
books, ratings, tags = load_and_clean_data(
    os.path.join('data', 'books.csv'),
    os.path.join('data', 'ratings.csv'),
    os.path.join('data', 'tags.csv'))

print("Building feature matrix...")
feature_matrix, tfidf, scaler = create_feature_matrix(books)

print("Training KNN...")
knn = KNNRecommender()
knn.fit(books, feature_matrix, tfidf=tfidf)   # pass tfidf for keyword search

print("Training SVD...")
svd = SVDRecommender()
svd.fit(ratings, books)

hybrid = HybridRecommender(knn, svd, books, feature_matrix)
print("✅ Ready — http://localhost:5000")
# ───────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📚 Book Recommender</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0d0d1a;color:#e2e8f0;min-height:100vh}

/* ── Header ── */
.header{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:44px 20px;text-align:center;border-bottom:1px solid rgba(99,179,237,.18)}
.header h1{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#63b3ed,#b794f4,#f6ad55);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
.header p{color:#94a3b8;font-size:.95rem}

/* ── Container ── */
.container{max-width:960px;margin:40px auto;padding:0 20px}

/* ── Search Card ── */
.search-card{background:linear-gradient(145deg,#1e1e3a,#1a2040);border:1px solid rgba(99,179,237,.18);border-radius:20px;padding:32px;margin-bottom:28px;box-shadow:0 20px 60px rgba(0,0,0,.4)}
.search-card h2{font-size:.95rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:20px;font-weight:500}

/* mode pills */
.mode-pills{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}
.pill{padding:6px 14px;border-radius:20px;font-size:.78rem;font-weight:600}
.pill-blue  {background:rgba(99,179,237,.13);color:#63b3ed;border:1px solid rgba(99,179,237,.3)}
.pill-purple{background:rgba(183,148,244,.13);color:#b794f4;border:1px solid rgba(183,148,244,.3)}
.pill-orange{background:rgba(246,173,85,.13);color:#f6ad55;border:1px solid rgba(246,173,85,.3)}
.pill-green {background:rgba(104,211,145,.13);color:#68d391;border:1px solid rgba(104,211,145,.3)}

/* inputs */
.row-inputs{display:grid;grid-template-columns:1fr 160px 120px;gap:12px;margin-bottom:16px}
@media(max-width:640px){.row-inputs{grid-template-columns:1fr}}
.fl{position:relative}
.fl label{display:block;font-size:.8rem;color:#94a3b8;margin-bottom:7px;font-weight:500}
.fl input{width:100%;padding:13px 16px;background:rgba(255,255,255,.05);border:1px solid rgba(99,179,237,.2);border-radius:12px;color:#e2e8f0;font-size:.95rem;font-family:'Inter',sans-serif;outline:none;transition:all .2s}
.fl input:focus{border-color:#63b3ed;background:rgba(99,179,237,.07);box-shadow:0 0 0 3px rgba(99,179,237,.13)}
.fl input::placeholder{color:#4a5568}

/* search button */
.btn-search{width:100%;padding:15px;background:linear-gradient(135deg,#4c51bf,#6b46c1);border:none;border-radius:12px;color:#fff;font-size:1rem;font-weight:600;font-family:'Inter',sans-serif;cursor:pointer;transition:all .25s;letter-spacing:.02em}
.btn-search:hover{background:linear-gradient(135deg,#5a67d8,#805ad5);transform:translateY(-1px);box-shadow:0 8px 25px rgba(107,70,193,.4)}
.btn-search:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* ── Banner ── */
.banner{display:none;padding:12px 20px;border-radius:12px;margin-bottom:22px;font-size:.88rem;font-weight:500}
.banner.search  {background:rgba(104,211,145,.1);border:1px solid rgba(104,211,145,.3);color:#68d391}
.banner.discover{background:rgba(99,179,237,.1);border:1px solid rgba(99,179,237,.3);color:#63b3ed}
.banner.hybrid  {background:rgba(246,173,85,.1);border:1px solid rgba(246,173,85,.3);color:#f6ad55}

/* ── Book Detail Card (seed book) ── */
#detail-card{display:none;background:linear-gradient(145deg,#1a2540,#1a2040);border:1px solid rgba(104,211,145,.25);border-radius:18px;padding:26px 28px;margin-bottom:28px;position:relative;overflow:hidden}
#detail-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#68d391,#4299e1,#b794f4)}
.detail-label{font-size:.72rem;color:#68d391;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:10px}
.detail-title{font-size:1.35rem;font-weight:700;color:#f0f4ff;margin-bottom:6px;line-height:1.3}
.detail-author{font-size:.9rem;color:#94a3b8;margin-bottom:8px}
.detail-meta{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.meta-chip{font-size:.78rem;padding:4px 12px;border-radius:16px;font-weight:500}
.chip-star  {background:rgba(246,173,85,.12);color:#f6ad55;border:1px solid rgba(246,173,85,.25)}
.chip-genre {background:rgba(183,148,244,.12);color:#b794f4;border:1px solid rgba(183,148,244,.25)}
.chip-count {background:rgba(99,179,237,.12);color:#63b3ed;border:1px solid rgba(99,179,237,.25)}

/* ── Results ── */
#results{display:none}
.section-title{font-size:.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:500;margin-bottom:16px}
.book-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}

.book-card{background:linear-gradient(145deg,#1e1e3a,#1a2040);border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:20px;transition:all .25s;animation:fadeUp .4s ease both;cursor:default}
.book-card:hover{border-color:rgba(99,179,237,.3);transform:translateY(-3px);box-shadow:0 12px 35px rgba(0,0,0,.4)}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.card-rank{font-size:.72rem;color:#4a5568;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:9px}
.card-title{font-size:.93rem;font-weight:600;color:#e2e8f0;line-height:1.4;margin-bottom:6px}
.card-author{font-size:.78rem;color:#94a3b8;margin-bottom:4px}
.card-genre{font-size:.73rem;color:#a0aec0;margin-bottom:12px}
.card-footer{display:flex;justify-content:space-between;align-items:center}
.star-lbl{font-size:.78rem;color:#f6ad55}
.score-pill{font-size:.73rem;font-weight:600;padding:3px 10px;border-radius:16px;background:rgba(99,179,237,.1);color:#63b3ed;border:1px solid rgba(99,179,237,.2)}

/* ── Spinner ── */
.spinner{display:none;text-align:center;padding:50px}
.ring{width:48px;height:48px;border:4px solid rgba(99,179,237,.2);border-top-color:#63b3ed;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner p{color:#94a3b8;font-size:.88rem}

/* error */
.err{display:none;padding:14px 20px;background:rgba(252,129,129,.1);border:1px solid rgba(252,129,129,.3);border-radius:12px;color:#fc8181;font-size:.88rem;margin-bottom:18px}
</style>
</head>
<body>

<div class="header">
  <h1>📚 Book Recommender</h1>
  <p>Hybrid KNN + SVD · Goodbooks-10k · 5.9M ratings · Smart Intent Detection</p>
</div>

<div class="container">

  <!-- Search Card -->
  <div class="search-card">
    <h2>Smart Book Search</h2>

    <div class="mode-pills">
      <span class="pill pill-green">🔍 Exact title → Book Details + Similar</span>
      <span class="pill pill-blue">💡 "fantasy books" → Discovery mode</span>
      <span class="pill pill-purple">🧊 No user → KNN only</span>
      <span class="pill pill-orange">🔥 With user → Hybrid KNN+SVD</span>
    </div>

    <div class="row-inputs">
      <div class="fl">
        <label>Search query</label>
        <input id="q" type="text"
               placeholder='Book title, genre, or "a book like Harry Potter"…' />
      </div>
      <div class="fl">
        <label>User ID <span style="color:#4a5568">(optional)</span></label>
        <input id="uid" type="number" placeholder="e.g. 12874" />
      </div>
      <div class="fl">
        <label>Results</label>
        <input id="topk" type="number" value="10" min="1" max="20" />
      </div>
    </div>

    <button class="btn-search" id="btn" onclick="go()">🔍 Search</button>
  </div>

  <div class="err" id="err"></div>
  <div class="banner" id="banner"></div>

  <!-- Book Detail (only in Search mode) -->
  <div id="detail-card">
    <div class="detail-label">📖 Book Found</div>
    <div class="detail-title" id="d-title"></div>
    <div class="detail-author" id="d-author"></div>
    <div class="detail-meta">
      <span class="meta-chip chip-star"  id="d-rating"></span>
      <span class="meta-chip chip-count" id="d-count"></span>
      <span class="meta-chip chip-genre" id="d-genre"></span>
    </div>
  </div>

  <!-- Spinner -->
  <div class="spinner" id="spinner">
    <div class="ring"></div>
    <p>Searching...</p>
  </div>

  <!-- Results -->
  <div id="results">
    <div class="section-title" id="res-label">Recommendations</div>
    <div class="book-grid" id="grid"></div>
  </div>

</div>

<script>
async function go() {
  const query = document.getElementById('q').value.trim();
  const uid   = document.getElementById('uid').value.trim();
  const topk  = parseInt(document.getElementById('topk').value) || 10;

  hideErr(); resetResults();
  if (!query) { showErr('Please enter a search query.'); return; }

  setLoading(true);
  try {
    const body = { query, top_k: topk };
    if (uid) body.user_id = parseInt(uid);

    const res  = await fetch('/smart-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.error) { showErr(data.error); return; }

    // Banner
    const banner = document.getElementById('banner');
    banner.className = 'banner ' + data.mode;
    if (data.mode === 'search') {
      banner.textContent = `🔍 Search Mode — Found "${data.book_detail?.title}". Showing similar books below.`;
    } else if (data.detected_genre) {
      banner.textContent = `🏷️ Genre Filter — Showing top-rated "${data.detected_genre}" books`;
      banner.className = 'banner discover';
    } else {
      banner.textContent = `💡 Discovery Mode — Showing top matches for "${query}"`;
    }
    banner.style.display = 'block';

    // Detail card (Search mode only)
    if (data.mode === 'search' && data.book_detail) {
      const d = data.book_detail;
      document.getElementById('d-title').textContent  = d.title;
      document.getElementById('d-author').textContent = '✍️ ' + cleanStr(d.authors);
      document.getElementById('d-rating').textContent = '⭐ ' + d.average_rating.toFixed(2);
      document.getElementById('d-count').textContent  = '📊 ' + (d.ratings_count || '').toLocaleString() + ' ratings';
      document.getElementById('d-genre').textContent  = '🏷️ ' + trimGenres(d.genres);
      document.getElementById('detail-card').style.display = 'block';
    }

    // Book cards
    const label = data.mode === 'search'
      ? `You might also like — based on "${data.book_detail?.title}"`
      : `Top ${data.recommendations.length} results for "${query}"`;
    document.getElementById('res-label').textContent = label;

    const grid = document.getElementById('grid');
    data.recommendations.forEach((bk, i) => {
      const card = document.createElement('div');
      card.className = 'book-card';
      card.style.animationDelay = `${i * 0.05}s`;
      const score = bk.score != null ? (bk.score * 100).toFixed(1) + '%' : '';
      const genres = trimGenres(bk.genres || '');
      card.innerHTML = `
        <div class="card-rank">#${i+1}</div>
        <div class="card-title">${esc(bk.title)}</div>
        <div class="card-author">✍️ ${esc(cleanStr(bk.authors || ''))}</div>
        <div class="card-genre">🏷️ ${esc(genres || 'Uncategorized')}</div>
        <div class="card-footer">
          <span class="star-lbl">⭐ ${(bk.average_rating||0).toFixed(2)}</span>
          ${score ? `<span class="score-pill">${score}</span>` : ''}
        </div>`;
      grid.appendChild(card);
    });

    document.getElementById('results').style.display = 'block';
  } catch(e) {
    showErr('Server error: ' + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Helpers ──────────────────────────────────────────────────────
function cleanStr(s){ return s.replace(/[\[\]']/g,'').substring(0,45) }
function trimGenres(s){ return s.replace(/[\[\]']/g,'').split(',').slice(0,2).join(', ') }
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML }
function showErr(m){ const e=document.getElementById('err'); e.textContent='⚠️ '+m; e.style.display='block' }
function hideErr(){ document.getElementById('err').style.display='none' }
function resetResults(){
  document.getElementById('results').style.display='none';
  document.getElementById('detail-card').style.display='none';
  document.getElementById('banner').style.display='none';
  document.getElementById('grid').innerHTML='';
}
function setLoading(on){
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btn').disabled = on;
}
document.addEventListener('keydown', e => { if(e.key==='Enter') go(); });
</script>
</body></html>"""


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return HTML


@app.route('/smart-search', methods=['POST'])
def smart_search():
    body    = request.get_json()
    query   = body.get('query', '').strip()
    user_id = body.get('user_id', None)
    top_k   = int(body.get('top_k', 10))

    if not query:
        return jsonify({'error': 'query is required'}), 400

    # ── Intent Detection ─────────────────────────────────────────────────────
    exact = knn.exact_lookup(query)   # None if no book title matches
    mode  = 'search' if exact is not None else 'discover'

    # ── Search Mode ───────────────────────────────────────────────────────────
    if mode == 'search':
        book_detail = knn.get_book_detail(query)

        # Get recommendations for the matched book (Hybrid if user provided)
        try:
            recs = hybrid.recommend(
                book_title=exact['title'],
                user_id=user_id,
                ratings_df=ratings if user_id else None,
                top_k=top_k)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        if recs is None or recs.empty:
            return jsonify({'error': f'No recommendations found for "{exact["title"]}"'}), 404

        # Strategy banner label
        if user_id:
            cnt = ratings[ratings['user_id'] == user_id].shape[0]
            strategy = ('hybrid' if cnt >= hybrid.cold_start_threshold else 'knn')
        else:
            strategy = 'search'

        score_col = 'final_score' if 'final_score' in recs.columns else 'similarity_score'
        items = [_book_dict(row, score_col) for _, row in recs.iterrows()]
        return jsonify({'mode': mode, 'strategy': strategy,
                        'book_detail': book_detail, 'recommendations': items})

    # ── Discovery Mode ────────────────────────────────────────────────────────
    # Check if query has genre intent → knn.search_by_keyword handles routing internally
    detected_genre = knn._extract_genre_from_query(query)
    recs = knn.search_by_keyword(query, top_k=top_k)
    if recs.empty:
        return jsonify({'error': f'No books found for "{query}"'}), 404

    items = [_book_dict(row, 'similarity_score') for _, row in recs.iterrows()]
    # Tell the UI which sub-strategy was used so the banner is informative
    sub_strategy = f'genre:{detected_genre}' if detected_genre else 'tfidf'
    return jsonify({'mode': mode, 'strategy': sub_strategy,
                    'detected_genre': detected_genre,
                    'book_detail': None, 'recommendations': items})


def _book_dict(row, score_col):
    d = {
        'title':          str(row.get('title', '')),
        'authors':        str(row.get('authors', '')),
        'genres':         str(row.get('genres', '')),
        'average_rating': float(row.get('average_rating', 0)),
    }
    if score_col and score_col in row.index:
        d['score'] = float(row[score_col])
    return d


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
