import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split


def load_data(ratings_path="data/ratings.csv", items_path="data/items.csv"):
    ratings = pd.read_csv(ratings_path)
    items = pd.read_csv(items_path)
    return ratings, items


def build_content_similarity(items):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(items["features"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim


def train_collaborative_model(train_ratings, test_ratings=None):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(train_ratings[["user_id", "item_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()
    model = SVD(random_state=42)
    model.fit(trainset)

    rmse = None
    if test_ratings is not None:
        testset = list(zip(test_ratings["user_id"], test_ratings["item_id"], test_ratings["rating"]))
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
    return model, rmse


def user_profile_score(user_id, item_id, ratings, items, cosine_sim):
    item_ids = items["item_id"].tolist()
    if item_id not in item_ids:
        return 0.0
    item_index = item_ids.index(item_id)

    user_rated = ratings[ratings["user_id"] == user_id]
    if user_rated.empty:
        return cosine_sim[item_index].mean()

    sims, weights = [], []
    for _, row in user_rated.iterrows():
        rated_item_id = row["item_id"]
        if rated_item_id not in item_ids or rated_item_id == item_id:
            continue
        rated_index = item_ids.index(rated_item_id)
        sim = cosine_sim[item_index][rated_index]
        sims.append(sim * row["rating"])
        weights.append(row["rating"])

    if not weights or sum(weights) == 0:
        return cosine_sim[item_index].mean()

    return sum(sims) / sum(weights)


def hybrid_score(user_id, item_id, model, ratings, items, cosine_sim, collab_weight=0.7, content_weight=0.3):
    try:
        collab_score = model.predict(user_id, item_id).est
    except Exception:
        collab_score = ratings["rating"].mean()

    content_raw = user_profile_score(user_id, item_id, ratings, items, cosine_sim)
    content_score = 1 + content_raw * 4

    return (collab_weight * collab_score) + (content_weight * content_score)


def is_cold_start_user(user_id, ratings, threshold=3):
    return ratings[ratings["user_id"] == user_id].shape[0] < threshold


def recommend_items(user_id, model, ratings, items, cosine_sim, top_n=5, collab_weight=0.7, content_weight=0.3):
    already_rated = set(ratings[ratings["user_id"] == user_id]["item_id"].tolist())
    candidates = items[~items["item_id"].isin(already_rated)]

    if is_cold_start_user(user_id, ratings):
        avg_item_scores = ratings.groupby("item_id")["rating"].mean()
        scored = candidates.copy()
        scored["score"] = scored["item_id"].map(avg_item_scores).fillna(ratings["rating"].mean())
        top = scored.sort_values("score", ascending=False).head(top_n)
        return list(zip(top["item_id"], top["score"])), True

    scores = []
    for item_id in candidates["item_id"]:
        score = hybrid_score(user_id, item_id, model, ratings, items, cosine_sim, collab_weight, content_weight)
        scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n], False


def precision_at_k(model, train_ratings, test_ratings, items, cosine_sim, k=5, rating_threshold=4,
                   collab_weight=0.7, content_weight=0.3):
    """Recommend using only TRAIN ratings; a hit = recommended item the user rated >= threshold in TEST."""
    precisions = []
    for user_id in test_ratings["user_id"].unique():
        user_test = test_ratings[test_ratings["user_id"] == user_id]
        relevant_items = set(user_test[user_test["rating"] >= rating_threshold]["item_id"])
        if not relevant_items:
            continue
        recs, _ = recommend_items(user_id, model, train_ratings, items, cosine_sim, top_n=k,
                                  collab_weight=collab_weight, content_weight=content_weight)
        recommended_items = set(r[0] for r in recs)
        hits = len(recommended_items & relevant_items)
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0


def save_model(model, cosine_sim, items, path="hybrid_model.pkl"):
    joblib.dump({"collab_model": model, "cosine_sim": cosine_sim, "items": items}, path)


def load_model(path="hybrid_model.pkl"):
    return joblib.load(path)


def print_recommendations(user_id, recs, cold_start):
    tag = " (cold-start: using popularity fallback)" if cold_start else ""
    print(f"\nRecommended Items for User {user_id}{tag}:")
    for item_id, score in recs:
        print(f"- Item {item_id} -> {score:.2f}")


def main():
    ratings, items = load_data()
    cosine_sim = build_content_similarity(items)

    # Evaluation: hold out 20% of ratings
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=42)
    eval_model, rmse = train_collaborative_model(train_r, test_r)
    print(f"Collaborative Filtering RMSE: {rmse:.4f}")

    p_at_5 = precision_at_k(eval_model, train_r, test_r, items, cosine_sim, k=5)
    print(f"Precision@5 (hybrid): {p_at_5:.4f}")

    # Final model: train on all ratings for real recommendations
    model, _ = train_collaborative_model(ratings)
    save_model(model, cosine_sim, items)
    print("Model saved to hybrid_model.pkl")

    while True:
        user_input = input("\nEnter user_id to get recommendations (or 'quit'): ").strip()
        if user_input.lower() == "quit":
            break
        try:
            user_id = int(user_input)
        except ValueError:
            print("Invalid user_id.")
            continue
        recs, cold_start = recommend_items(user_id, model, ratings, items, cosine_sim, top_n=5)
        print_recommendations(user_id, recs, cold_start)


if __name__ == "__main__":
    main()
