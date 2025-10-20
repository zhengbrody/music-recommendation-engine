"""Collaborative filtering models using implicit and surprise libraries."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
from pathlib import Path


class ImplicitALSModel:
    """Matrix Factorization using Alternating Least Squares (implicit library)."""

    def __init__(self, factors=50, regularization=0.01, iterations=10):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        self.user_items_sparse = None
        self.user_id_map = None
        self.song_id_map = None

    def prepare_data(self, interactions_df, user_id_map, song_id_map):
        """Prepare sparse matrix for training."""
        self.user_id_map = user_id_map
        self.song_id_map = song_id_map

        # Create reverse mappings
        idx_to_user = {v: k for k, v in user_id_map.items()}
        idx_to_song = {v: k for k, v in song_id_map.items()}

        # Create sparse matrix
        n_users = len(user_id_map)
        n_songs = len(song_id_map)

        row = interactions_df['user_idx'].values
        col = interactions_df['song_idx'].values
        data = interactions_df['rating'].values

        self.user_items_sparse = csr_matrix(
            (data, (row, col)),
            shape=(n_users, n_songs)
        )

        return self.user_items_sparse

    def train(self):
        """Train the ALS model."""
        print("Training Implicit ALS model...")
        self.model.fit(self.user_items_sparse)
        print("Training complete!")

    def recommend(self, user_id, n=10, filter_already_liked=True):
        """Get recommendations for a user.

        Args:
            user_id: Original user ID
            n: Number of recommendations
            filter_already_liked: Whether to filter out already interacted items

        Returns:
            List of (song_id, score) tuples
        """
        if user_id not in self.user_id_map:
            return []

        user_idx = self.user_id_map[user_id]
        idx_to_song = {v: k for k, v in self.song_id_map.items()}

        # Get recommendations
        song_indices, scores = self.model.recommend(
            user_idx,
            self.user_items_sparse[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )

        recommendations = [
            (idx_to_song[int(song_idx)], float(score))
            for song_idx, score in zip(song_indices, scores)
        ]

        return recommendations

    def save(self, filepath):
        """Save model to disk."""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_items_sparse': self.user_items_sparse,
                'user_id_map': self.user_id_map,
                'song_id_map': self.song_id_map
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.user_items_sparse = data['user_items_sparse']
            self.user_id_map = data['user_id_map']
            self.song_id_map = data['song_id_map']
        print(f"Model loaded from {filepath}")


class SurpriseSVDModel:
    """SVD-based collaborative filtering using Surprise library."""

    def __init__(self, n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.trainset = None
        self.all_song_ids = None

    def prepare_data(self, interactions_df):
        """Prepare data for Surprise library."""
        # Surprise requires specific format
        reader = Reader(rating_scale=(interactions_df['rating'].min(),
                                     interactions_df['rating'].max()))

        data = Dataset.load_from_df(
            interactions_df[['user_id', 'song_id', 'rating']],
            reader
        )

        self.trainset = data.build_full_trainset()
        self.all_song_ids = interactions_df['song_id'].unique()

        return self.trainset

    def train(self):
        """Train the SVD model."""
        print("Training Surprise SVD model...")
        self.model.fit(self.trainset)
        print("Training complete!")

    def recommend(self, user_id, n=10, interactions_df=None):
        """Get recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            interactions_df: DataFrame to filter already interacted items

        Returns:
            List of (song_id, predicted_rating) tuples
        """
        # Get songs the user hasn't interacted with
        if interactions_df is not None:
            user_songs = set(interactions_df[interactions_df['user_id'] == user_id]['song_id'])
            candidate_songs = [s for s in self.all_song_ids if s not in user_songs]
        else:
            candidate_songs = self.all_song_ids

        # Predict ratings
        predictions = [
            (song_id, self.model.predict(user_id, song_id).est)
            for song_id in candidate_songs
        ]

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

    def save(self, filepath):
        """Save model to disk."""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trainset': self.trainset,
                'all_song_ids': self.all_song_ids
            }, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.trainset = data['trainset']
            self.all_song_ids = data['all_song_ids']
        print(f"Model loaded from {filepath}")
