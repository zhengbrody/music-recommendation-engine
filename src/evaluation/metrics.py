"""
Evaluation metrics for recommendation systems.
Implements Precision@K, Recall@K, NDCG@K, MAP, Coverage, etc.
"""

import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class RecommenderMetrics:
    """Calculate evaluation metrics for recommendation systems."""

    @staticmethod
    def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate Precision@K.

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            Precision@K score
        """
        if k == 0 or len(recommended) == 0:
            return 0.0

        recommended_k = set(recommended[:k])
        n_relevant_and_recommended = len(recommended_k & relevant)

        return n_relevant_and_recommended / k

    @staticmethod
    def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate Recall@K.

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0

        recommended_k = set(recommended[:k])
        n_relevant_and_recommended = len(recommended_k & relevant)

        return n_relevant_and_recommended / len(relevant)

    @staticmethod
    def f1_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate F1-Score@K.

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            F1@K score
        """
        precision = RecommenderMetrics.precision_at_k(recommended, relevant, k)
        recall = RecommenderMetrics.recall_at_k(recommended, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(recommended: List[int], relevant: Set[int]) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices

        Returns:
            Average Precision score
        """
        if len(relevant) == 0:
            return 0.0

        score = 0.0
        n_hits = 0

        for k, item in enumerate(recommended, 1):
            if item in relevant:
                n_hits += 1
                score += n_hits / k

        return score / len(relevant)

    @staticmethod
    def mean_average_precision(
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).

        Args:
            recommendations: Dict mapping user_idx to list of recommended items
            ground_truth: Dict mapping user_idx to set of relevant items

        Returns:
            MAP score
        """
        aps = []
        for user_idx in recommendations:
            if user_idx in ground_truth:
                ap = RecommenderMetrics.average_precision(
                    recommendations[user_idx],
                    ground_truth[user_idx]
                )
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    @staticmethod
    def dcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@K.

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, item in enumerate(recommended[:k], 1):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score
        """
        dcg = RecommenderMetrics.dcg_at_k(recommended, relevant, k)

        # Calculate IDCG (ideal DCG)
        ideal_k = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Calculate Hit Rate@K (binary: 1 if any hit, 0 otherwise).

        Args:
            recommended: List of recommended item indices
            relevant: Set of relevant (ground truth) item indices
            k: Number of top recommendations to consider

        Returns:
            Hit rate (0 or 1)
        """
        recommended_k = set(recommended[:k])
        return 1.0 if len(recommended_k & relevant) > 0 else 0.0

    @staticmethod
    def coverage(
        recommendations: Dict[int, List[int]],
        n_items: int
    ) -> float:
        """
        Calculate catalog coverage (percentage of items recommended).

        Args:
            recommendations: Dict mapping user_idx to list of recommended items
            n_items: Total number of items in catalog

        Returns:
            Coverage score (0 to 1)
        """
        recommended_items = set()
        for recs in recommendations.values():
            recommended_items.update(recs)

        return len(recommended_items) / n_items

    @staticmethod
    def diversity(recommendations: Dict[int, List[int]]) -> float:
        """
        Calculate recommendation diversity (average unique items per user).

        Args:
            recommendations: Dict mapping user_idx to list of recommended items

        Returns:
            Average diversity score
        """
        diversities = [len(set(recs)) / len(recs) for recs in recommendations.values() if len(recs) > 0]
        return np.mean(diversities) if diversities else 0.0

    @staticmethod
    def novelty(
        recommendations: Dict[int, List[int]],
        item_popularity: Dict[int, int]
    ) -> float:
        """
        Calculate novelty (how unpopular recommended items are).

        Args:
            recommendations: Dict mapping user_idx to list of recommended items
            item_popularity: Dict mapping item_idx to popularity count

        Returns:
            Average novelty score
        """
        total_popularity = sum(item_popularity.values())
        novelties = []

        for recs in recommendations.values():
            for item in recs:
                if item in item_popularity:
                    # Novelty is inverse of popularity
                    novelty = -np.log2(item_popularity[item] / total_popularity)
                    novelties.append(novelty)

        return np.mean(novelties) if novelties else 0.0

    @staticmethod
    def evaluate_all(
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        k: int = 10,
        n_items: Optional[int] = None,
        item_popularity: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.

        Args:
            recommendations: Dict mapping user_idx to list of recommended items
            ground_truth: Dict mapping user_idx to set of relevant items
            k: Number of top recommendations to consider
            n_items: Total number of items (for coverage)
            item_popularity: Item popularity counts (for novelty)

        Returns:
            Dictionary with all metric scores
        """
        metrics = {}

        # Ranking metrics (average across users)
        precisions = []
        recalls = []
        f1s = []
        ndcgs = []
        hit_rates = []

        for user_idx in recommendations:
            if user_idx in ground_truth:
                recommended = recommendations[user_idx]
                relevant = ground_truth[user_idx]

                precisions.append(RecommenderMetrics.precision_at_k(recommended, relevant, k))
                recalls.append(RecommenderMetrics.recall_at_k(recommended, relevant, k))
                f1s.append(RecommenderMetrics.f1_at_k(recommended, relevant, k))
                ndcgs.append(RecommenderMetrics.ndcg_at_k(recommended, relevant, k))
                hit_rates.append(RecommenderMetrics.hit_rate_at_k(recommended, relevant, k))

        metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'f1@{k}'] = np.mean(f1s) if f1s else 0.0
        metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0

        # MAP
        metrics['map'] = RecommenderMetrics.mean_average_precision(recommendations, ground_truth)

        # Coverage
        if n_items:
            metrics['coverage'] = RecommenderMetrics.coverage(recommendations, n_items)

        # Diversity
        metrics['diversity'] = RecommenderMetrics.diversity(recommendations)

        # Novelty
        if item_popularity:
            metrics['novelty'] = RecommenderMetrics.novelty(recommendations, item_popularity)

        return metrics
