"""
Content-Based Movie Recommendation System
===========================================

Builds per-user taste profiles from movie tag data and rated history,
then predicts ratings and generates Top-N recommendations using
cosine similarity between user profiles and movie content vectors.

Evaluation metrics:
    - MAE (Mean Absolute Error): how close predicted ratings are to actual ratings
    - Hit Ratio @ N: how often a user's held-out liked movie appears in their
      top-N recommendation list

Usage:
    python recommender.py --data_dir path/to/dataset --min_ratings 20 \
        --sample_users 500 --test_size 0.2 --top_n 10

Expected data directory structure:
    <data_dir>/
        scores/
            glmer.csv        # long-format: item_id, tag, score
        raw/
            ratings.json     # one JSON object per line: user_id, item_id, rating
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("--data_dir", required=True, help="Path to the movie dataset directory")
    parser.add_argument("--min_ratings", type=int, default=20, help="Minimum number of ratings per user")
    parser.add_argument("--sample_users", type=int, default=500, help="Limit number of users for testing")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of ratings held out for testing")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations per user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_content_matrix(data_dir):
    """Load and pivot the movie-tag score file into a wide item x tag matrix."""
    scores_path = os.path.join(data_dir, "scores", "glmer.csv")
    print(f"\nReading content scores from: {scores_path}")

    scores = pd.read_csv(scores_path)
    content_matrix = scores.pivot(index="item_id", columns="tag", values="score").fillna(0)
    print(f"Movies with content vectors: {content_matrix.shape[0]} | Tags: {content_matrix.shape[1]}")
    return content_matrix


def get_active_users(ratings_path, min_ratings, sample_users, seed):
    """Stream ratings.json once to count ratings per user and sample a subset of active users."""
    print("Counting ratings per user (streaming ratings.json)...")
    user_counts = {}
    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting"):
            r = json.loads(line)
            user_counts[r["user_id"]] = user_counts.get(r["user_id"], 0) + 1

    active_users = [u for u, c in user_counts.items() if c >= min_ratings]

    rng = np.random.default_rng(seed)
    rng.shuffle(active_users)
    if sample_users and len(active_users) > sample_users:
        active_users = active_users[:sample_users]

    print(f"Active users (>= {min_ratings} ratings): {len(active_users)}")
    return active_users


def load_filtered_ratings(ratings_path, active_users, content_matrix):
    """Stream ratings.json again, keeping only active users and known movies."""
    active_set = set(active_users)
    filtered = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Filtering"):
            r = json.loads(line)
            if r["user_id"] in active_set:
                filtered.append((r["user_id"], r["item_id"], r["rating"]))

    ratings_df = pd.DataFrame(filtered, columns=["user_id", "item_id", "rating"])
    ratings_df = ratings_df[ratings_df["item_id"].isin(content_matrix.index)]
    print(f"Filtered ratings: {ratings_df.shape}")
    return ratings_df


def build_user_profile(user_id, ratings_df, content_matrix):
    """Build a user's taste profile as the rating-weighted average of their movies' tag vectors."""
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    user_content = content_matrix.loc[user_ratings["item_id"]]
    weights = user_ratings["rating"].values
    return np.average(user_content, axis=0, weights=weights)


def predict_rating(user_id, item_id, user_profiles, content_matrix):
    """Predict a rating as cosine similarity between user profile and movie vector, scaled to 0-5."""
    if user_id not in user_profiles or item_id not in content_matrix.index:
        return np.nan
    sim = cosine_similarity(
        user_profiles[user_id].reshape(1, -1),
        content_matrix.loc[item_id].values.reshape(1, -1),
    )[0, 0]
    return sim * 5


def get_top_n_recommendations(user_id, train_df, content_matrix, user_profiles, n=10):
    """Rank all unseen movies for a user by cosine similarity to their profile and return the top N."""
    user_seen = set(train_df[train_df["user_id"] == user_id]["item_id"])
    candidates = [m for m in content_matrix.index if m not in user_seen]

    sims = cosine_similarity(
        user_profiles[user_id].reshape(1, -1),
        content_matrix.loc[candidates].values,
    )[0]

    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:n]]


def main():
    args = parse_args()

    # Step 1-2: Load movie content (tag) vectors
    content_matrix = load_content_matrix(args.data_dir)

    # Step 3-4: Identify active users and load their filtered ratings
    ratings_path = os.path.join(args.data_dir, "raw", "ratings.json")
    active_users = get_active_users(ratings_path, args.min_ratings, args.sample_users, args.seed)
    ratings_df = load_filtered_ratings(ratings_path, active_users, content_matrix)

    # Step 5: Train / test split
    train_df, test_df = train_test_split(ratings_df, test_size=args.test_size, random_state=args.seed)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Step 6: Build a taste profile for each active user from their training ratings
    print("Building user profiles...")
    user_profiles = {
        u: build_user_profile(u, train_df, content_matrix)
        for u in tqdm(active_users, desc="Building user profiles")
    }

    # Step 7: Predict held-out test ratings and compute MAE
    print("\nPredicting test ratings...")
    test_df = test_df.copy()
    test_df["predicted"] = [
        predict_rating(u, i, user_profiles, content_matrix)
        for u, i in tqdm(zip(test_df["user_id"], test_df["item_id"]), total=len(test_df))
    ]
    mae = mean_absolute_error(test_df["rating"], test_df["predicted"])
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")

    # Step 8: Evaluate Top-N Hit Ratio
    print("\nEvaluating Top-N hit ratio...")
    hits, users_evaluated = 0, 0
    for u in tqdm(active_users, desc="Top-N Evaluation"):
        true_items = set(test_df[test_df["user_id"] == u]["item_id"])
        if not true_items:
            continue
        recommended = get_top_n_recommendations(u, train_df, content_matrix, user_profiles, args.top_n)
        if any(item in recommended for item in true_items):
            hits += 1
        users_evaluated += 1

    hit_ratio = hits / users_evaluated if users_evaluated else 0
    print(f"Hit Ratio (Top-{args.top_n}): {hit_ratio:.4f}")

    return mae, hit_ratio


if __name__ == "__main__":
    main()
