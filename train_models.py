"""Train all recommendation models."""
import sys
from pathlib import Path
import pandas as pd
import pickle

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR
from src.preprocessing.data_loader import MusicDataLoader
from src.models.collaborative_filtering import ImplicitALSModel, SurpriseSVDModel
from src.models.deep_learning import DeepRecommender


def train_all_models():
    """Train all recommendation models."""
    print("="*60)
    print("TRAINING MUSIC RECOMMENDATION MODELS")
    print("="*60)

    # Load processed data
    loader = MusicDataLoader(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    print("\nLoading processed data...")
    data = loader.load_processed_data()

    interactions_df = data['interactions']
    user_id_map = data['user_id_map']
    song_id_map = data['song_id_map']

    print(f"  - Users: {len(user_id_map)}")
    print(f"  - Songs: {len(song_id_map)}")
    print(f"  - Interactions: {len(interactions_df)}")

    # 1. Train Implicit ALS Model
    print("\n" + "="*60)
    print("1. Training Implicit ALS Model")
    print("="*60)
    als_model = ImplicitALSModel(factors=50, regularization=0.01, iterations=15)
    als_model.prepare_data(interactions_df, user_id_map, song_id_map)
    als_model.train()
    als_model.save(MODEL_DIR / 'als_model.pkl')

    # Test recommendations
    test_user = list(user_id_map.keys())[0]
    recs = als_model.recommend(test_user, n=5)
    print(f"\nSample recommendations for user {test_user}:")
    for song_id, score in recs[:3]:
        print(f"  - Song {song_id}: {score:.4f}")

    # 2. Train Surprise SVD Model
    print("\n" + "="*60)
    print("2. Training Surprise SVD Model")
    print("="*60)
    svd_model = SurpriseSVDModel(n_factors=50, n_epochs=15)
    svd_model.prepare_data(interactions_df)
    svd_model.train()
    svd_model.save(MODEL_DIR / 'svd_model.pkl')

    # Test recommendations
    recs = svd_model.recommend(test_user, n=5, interactions_df=interactions_df)
    print(f"\nSample recommendations for user {test_user}:")
    for song_id, score in recs[:3]:
        print(f"  - Song {song_id}: {score:.4f}")

    # 3. Train Deep Learning Model
    print("\n" + "="*60)
    print("3. Training Deep Learning Model")
    print("="*60)
    n_users = len(user_id_map)
    n_songs = len(song_id_map)

    dl_model = DeepRecommender(
        n_users=n_users,
        n_songs=n_songs,
        n_factors=50,
        hidden_layers=[128, 64, 32],
        learning_rate=0.001
    )
    dl_model.user_id_map = user_id_map
    dl_model.song_id_map = song_id_map

    # Shuffle data for training
    interactions_shuffled = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)

    history = dl_model.train(
        interactions_shuffled,
        epochs=10,
        batch_size=256,
        validation_split=0.2
    )
    dl_model.save(MODEL_DIR / 'deep_model.pt')

    # Test recommendations
    recs = dl_model.recommend(test_user, n=5, interactions_df=interactions_df)
    print(f"\nSample recommendations for user {test_user}:")
    for song_id, score in recs[:3]:
        print(f"  - Song {song_id}: {score:.4f}")

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved to: {MODEL_DIR}")
    print("  ✓ als_model.pkl")
    print("  ✓ svd_model.pkl")
    print("  ✓ deep_model.pt")
    print("\nYou can now run the API and dashboard:")
    print("  - API: python api/app.py")
    print("  - Dashboard: python dashboard/app.py")


if __name__ == "__main__":
    train_all_models()
