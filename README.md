# Hybrid Recommendation System (Content + Collaborative Filtering)

## Files
- `data/items.csv` — 50 items with genre-style features (item_id, title, features)
- `data/ratings.csv` — 455 synthetic ratings from 40 users (user_id, item_id, rating 1-5)
- `main.py` — single script: content-based filtering, collaborative filtering (SVD via Surprise),
  hybrid scoring, cold-start handling, RMSE + Precision@K evaluation, model save, CLI
- `requirements.txt`
- `hybrid_model.pkl` — saved model (generated after running)

## Run
```bash
pip install -r requirements.txt
python main.py
```

Prints RMSE and Precision@5, saves the trained model, then prompts for a `user_id` to
generate top-5 hybrid recommendations.

## How it works
- **Content-based**: TF-IDF over item `features`, cosine similarity between items. A user's
  content score for a candidate item is the rating-weighted average similarity to items
  they've already rated (their "profile").
- **Collaborative**: SVD matrix factorization (Surprise library) trained on the user-item
  rating matrix, predicting a rating for any (user, item) pair.
- **Hybrid**: `final_score = 0.7 * collaborative_score + 0.3 * content_score` (weights are
  parameters — pass `collab_weight` / `content_weight` to `recommend_items()` to adjust).
- **Cold start**: if a user has fewer than 3 ratings, the system skips the hybrid scoring
  (which would be unreliable) and falls back to recommending the highest-average-rated
  items overall (a popularity baseline).
- **Evaluation**: RMSE from Surprise's train/test split for the collaborative half;
  Precision@K computed by checking, for each user, how many of the top-K recommendations
  overlap with items they actually rated >= 4.

## Note on numbers
This dataset is synthetic and randomly generated (genre preferences + random noise), so the
RMSE and Precision@K values are only meant to demonstrate the pipeline runs correctly —
they are not representative of real-world recommender performance. Swap in a real dataset
(e.g. MovieLens) in `data/` for meaningful metrics.
