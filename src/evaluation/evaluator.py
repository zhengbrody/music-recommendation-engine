"""
Model evaluation framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional
from .metrics import RecommenderMetrics
from config.config import Config


class ModelEvaluator:
    """Framework for evaluating recommendation models."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize evaluator.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.metrics = RecommenderMetrics()

    def prepare_ground_truth(self, test_df: pd.DataFrame) -> Dict[int, Set[int]]:
        """
        Prepare ground truth from test DataFrame.

        Args:
            test_df: Test DataFrame with user_idx and artist_idx

        Returns:
            Dictionary mapping user_idx to set of relevant artist_idx
        """
        ground_truth = {}
        for user_idx in test_df['user_idx'].unique():
            user_test = test_df[test_df['user_idx'] == user_idx]
            ground_truth[user_idx] = set(user_test['artist_idx'].values)

        return ground_truth

    def evaluate_model(
        self,
        model,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_users: Optional[int] = None,
        n_artists: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a recommendation model.

        Args:
            model: Trained recommender model (ALS or NCF)
            test_df: Test DataFrame
            train_df: Training DataFrame (for filtering)
            n_users: Number of users to evaluate (None = all)
            n_artists: Total number of artists (for coverage)
            k: Top-K for evaluation (uses config default if None)

        Returns:
            Dictionary with evaluation metrics
        """
        k = k or self.config.TOP_K

        # Prepare ground truth
        ground_truth = self.prepare_ground_truth(test_df)

        # Get unique test users
        test_users = list(ground_truth.keys())
        if n_users:
            test_users = test_users[:n_users]

        print(f"Evaluating {len(test_users)} users with K={k}...")

        # Generate recommendations
        recommendations = {}
        for user_idx in test_users:
            try:
                # Get user's training interactions (to exclude)
                user_train = train_df[train_df['user_idx'] == user_idx]
                exclude_artists = set(user_train['artist_idx'].values) if len(user_train) > 0 else set()

                # Generate recommendations
                if hasattr(model, 'recommend_for_user'):
                    # ALS or NCF model
                    if hasattr(model, 'user_artist_matrix'):
                        # ALS model
                        recs = model.recommend_for_user(user_idx, n=k, filter_already_liked=True)
                    else:
                        # NCF model
                        recs = model.recommend_for_user(user_idx, n=k, exclude_artists=list(exclude_artists))

                    recommendations[user_idx] = [artist_idx for artist_idx, score in recs]
                else:
                    recommendations[user_idx] = []

            except Exception as e:
                print(f"Error evaluating user {user_idx}: {e}")
                recommendations[user_idx] = []

        # Calculate item popularity for novelty
        item_popularity = train_df.groupby('artist_idx').size().to_dict()

        # Evaluate all metrics
        metrics = self.metrics.evaluate_all(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k=k,
            n_items=n_artists,
            item_popularity=item_popularity
        )

        return metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted way.

        Args:
            metrics: Dictionary with metric scores
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:20s}: {value:.4f}")

        print("="*50 + "\n")

    def compare_models(
        self,
        models: Dict[str, any],
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        n_users: Optional[int] = None,
        n_artists: Optional[int] = None,
        k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.

        Args:
            models: Dictionary mapping model_name to model object
            test_df: Test DataFrame
            train_df: Training DataFrame
            n_users: Number of users to evaluate
            n_artists: Total number of artists
            k: Top-K for evaluation

        Returns:
            DataFrame with comparison results
        """
        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(
                model=model,
                test_df=test_df,
                train_df=train_df,
                n_users=n_users,
                n_artists=n_artists,
                k=k
            )
            results[model_name] = metrics

        # Convert to DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)

        return comparison_df
