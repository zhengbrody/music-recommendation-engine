"""
Alternating Least Squares (ALS) Collaborative Filtering Model.
Uses the implicit library for matrix factorization.
"""

import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from typing import List, Tuple, Optional, Dict
import pickle
from config.config import Config


class ALSRecommender:
    """ALS-based collaborative filtering recommender."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ALS recommender.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.model = AlternatingLeastSquares(
            factors=self.config.ALS_FACTORS,
            regularization=self.config.ALS_REGULARIZATION,
            iterations=self.config.ALS_ITERATIONS,
            random_state=self.config.RANDOM_STATE
        )
        self.user_artist_matrix = None
        self.trained = False

    def fit(self, interaction_matrix: sparse.csr_matrix) -> None:
        """
        Train the ALS model on interaction data.

        Args:
            interaction_matrix: Sparse CSR matrix of shape (n_users, n_artists)
        """
        print("Training ALS model...")
        print(f"Matrix shape: {interaction_matrix.shape}")
        print(f"Factors: {self.config.ALS_FACTORS}, "
              f"Regularization: {self.config.ALS_REGULARIZATION}, "
              f"Iterations: {self.config.ALS_ITERATIONS}")

        self.user_artist_matrix = interaction_matrix

        # implicit library expects (items x users) matrix
        # but we have (users x items), so we transpose
        artist_user_matrix = interaction_matrix.T.tocsr()

        # Fit the model
        self.model.fit(artist_user_matrix)
        self.trained = True

        print("ALS model training completed!")

    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 10,
        filter_already_liked: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_idx: User index
            n: Number of recommendations
            filter_already_liked: Whether to exclude already interacted items

        Returns:
            List of (artist_idx, score) tuples
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        if user_idx >= self.user_artist_matrix.shape[0]:
            raise ValueError(f"Invalid user_idx: {user_idx}")

        # Get recommendations
        artist_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_artist_matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )

        recommendations = list(zip(artist_ids, scores))
        return recommendations

    def recommend_for_users(
        self,
        user_indices: List[int],
        n: int = 10,
        filter_already_liked: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_indices: List of user indices
            n: Number of recommendations per user
            filter_already_liked: Whether to exclude already interacted items

        Returns:
            Dictionary mapping user_idx to list of (artist_idx, score) tuples
        """
        recommendations = {}
        for user_idx in user_indices:
            try:
                recs = self.recommend_for_user(user_idx, n, filter_already_liked)
                recommendations[user_idx] = recs
            except Exception as e:
                print(f"Error generating recommendations for user {user_idx}: {e}")
                recommendations[user_idx] = []

        return recommendations

    def similar_items(self, artist_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar artists to a given artist.

        Args:
            artist_idx: Artist index
            n: Number of similar artists

        Returns:
            List of (artist_idx, similarity_score) tuples
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Get similar items
        similar_ids, scores = self.model.similar_items(artist_idx, N=n + 1)

        # Remove the item itself (first result)
        similar_artists = list(zip(similar_ids[1:], scores[1:]))

        return similar_artists

    def similar_users(self, user_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar users to a given user.

        Args:
            user_idx: User index
            n: Number of similar users

        Returns:
            List of (user_idx, similarity_score) tuples
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Get similar users
        similar_ids, scores = self.model.similar_users(user_idx, N=n + 1)

        # Remove the user itself (first result)
        similar = list(zip(similar_ids[1:], scores[1:]))

        return similar

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """
        Get the learned embedding vector for a user.

        Args:
            user_idx: User index

        Returns:
            Embedding vector of shape (n_factors,)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        return self.model.user_factors[user_idx]

    def get_artist_embedding(self, artist_idx: int) -> np.ndarray:
        """
        Get the learned embedding vector for an artist.

        Args:
            artist_idx: Artist index

        Returns:
            Embedding vector of shape (n_factors,)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        return self.model.item_factors[artist_idx]

    def save_model(self, file_path: Optional[str] = None) -> None:
        """
        Save the trained model to disk.

        Args:
            file_path: Path to save model (uses config default if None)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        file_path = file_path or self.config.ALS_MODEL_FILE

        model_data = {
            'model': self.model,
            'user_artist_matrix': self.user_artist_matrix,
            'config': {
                'factors': self.config.ALS_FACTORS,
                'regularization': self.config.ALS_REGULARIZATION,
                'iterations': self.config.ALS_ITERATIONS
            }
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {file_path}")

    def load_model(self, file_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to load model from (uses config default if None)
        """
        file_path = file_path or self.config.ALS_MODEL_FILE

        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.user_artist_matrix = model_data['user_artist_matrix']
        self.trained = True

        print(f"Model loaded from: {file_path}")
