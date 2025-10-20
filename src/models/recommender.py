"""
Main Music Recommender Engine with Search Functionality.
Combines multiple recommendation algorithms and provides search capabilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from scipy import sparse
import pickle
import re
from difflib import SequenceMatcher
from .als import ALSRecommender
from .ncf import NCFRecommender
from config.config import Config


class MusicRecommender:
    """
    Unified music recommendation system with search functionality.
    Combines ALS and NCF models with artist/user search capabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Music Recommender.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.als_model = ALSRecommender(config)
        self.ncf_model = NCFRecommender(config)
        self.mappings = None
        self.train_df = None
        self.user_history = {}  # Cache of user interaction history
        self.artist_stats = {}  # Cache of artist statistics

    def load_data_and_models(self) -> None:
        """Load preprocessed data and trained models."""
        print("Loading data and models...")

        # Load mappings
        with open(self.config.MAPPINGS_FILE, 'rb') as f:
            self.mappings = pickle.load(f)

        # Load training data
        self.train_df = pd.read_csv(self.config.TRAIN_DATA_FILE)

        # Build user history cache
        self._build_user_history()

        # Build artist stats cache
        self._build_artist_stats()

        # Load ALS model
        self.als_model.load_model()

        # Load NCF model (optional)
        try:
            self.ncf_model.load_model()
        except FileNotFoundError:
            print("NCF model not found. Using ALS only.")

        print("Data and models loaded successfully!")

    def _build_user_history(self) -> None:
        """Build cache of user interaction history."""
        print("Building user history cache...")
        for user_idx in self.train_df['user_idx'].unique():
            user_data = self.train_df[self.train_df['user_idx'] == user_idx]
            self.user_history[user_idx] = set(user_data['artist_idx'].values)

    def _build_artist_stats(self) -> None:
        """Build cache of artist statistics."""
        print("Building artist statistics cache...")
        stats = self.train_df.groupby('artist_idx').agg({
            'play_count': ['sum', 'count', 'mean'],
            'user_idx': 'nunique'
        })

        for artist_idx in range(self.mappings['n_artists']):
            if artist_idx in stats.index:
                self.artist_stats[artist_idx] = {
                    'total_plays': int(stats.loc[artist_idx, ('play_count', 'sum')]),
                    'n_interactions': int(stats.loc[artist_idx, ('play_count', 'count')]),
                    'avg_plays': float(stats.loc[artist_idx, ('play_count', 'mean')]),
                    'n_users': int(stats.loc[artist_idx, ('user_idx', 'nunique')])
                }
            else:
                self.artist_stats[artist_idx] = {
                    'total_plays': 0,
                    'n_interactions': 0,
                    'avg_plays': 0.0,
                    'n_users': 0
                }

    # ==================== SEARCH FUNCTIONALITY ====================

    def search_artists(
        self,
        query: str,
        top_k: int = 10,
        method: str = 'fuzzy'
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Search for artists by name.

        Args:
            query: Search query string
            top_k: Number of results to return
            method: Search method ('exact', 'fuzzy', 'contains')

        Returns:
            List of dictionaries with artist info and match scores
        """
        query = query.strip().lower()
        results = []

        idx_to_artist = self.mappings['idx_to_artist']

        for artist_idx, artist_name in idx_to_artist.items():
            artist_lower = artist_name.lower()
            score = 0.0

            if method == 'exact':
                if artist_lower == query:
                    score = 1.0
            elif method == 'contains':
                if query in artist_lower:
                    score = len(query) / len(artist_lower)
            elif method == 'fuzzy':
                score = SequenceMatcher(None, query, artist_lower).ratio()
            else:
                raise ValueError(f"Unknown search method: {method}")

            if score > 0:
                results.append({
                    'artist_idx': artist_idx,
                    'artist_name': artist_name,
                    'match_score': score,
                    **self.artist_stats.get(artist_idx, {})
                })

        # Sort by score (descending) and take top-k
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:top_k]

    def search_users(
        self,
        user_id: str,
        method: str = 'exact'
    ) -> Optional[Dict[str, Union[int, str]]]:
        """
        Search for a user by ID.

        Args:
            user_id: User ID string
            method: Search method ('exact', 'contains')

        Returns:
            Dictionary with user info or None if not found
        """
        user_to_idx = self.mappings['user_to_idx']

        if method == 'exact':
            if user_id in user_to_idx:
                user_idx = user_to_idx[user_id]
                return {
                    'user_id': user_id,
                    'user_idx': user_idx,
                    'n_interactions': len(self.user_history.get(user_idx, set()))
                }
        elif method == 'contains':
            matches = [uid for uid in user_to_idx.keys() if user_id in uid]
            if matches:
                user_id = matches[0]
                user_idx = user_to_idx[user_id]
                return {
                    'user_id': user_id,
                    'user_idx': user_idx,
                    'n_interactions': len(self.user_history.get(user_idx, set()))
                }

        return None

    def search_by_popularity(self, top_k: int = 10) -> List[Dict]:
        """
        Get most popular artists by total plays.

        Args:
            top_k: Number of artists to return

        Returns:
            List of dictionaries with artist info
        """
        results = []
        for artist_idx, stats in self.artist_stats.items():
            if stats['total_plays'] > 0:
                results.append({
                    'artist_idx': artist_idx,
                    'artist_name': self.mappings['idx_to_artist'][artist_idx],
                    **stats
                })

        results.sort(key=lambda x: x['total_plays'], reverse=True)
        return results[:top_k]

    def search_by_user_count(self, top_k: int = 10) -> List[Dict]:
        """
        Get artists with most unique listeners.

        Args:
            top_k: Number of artists to return

        Returns:
            List of dictionaries with artist info
        """
        results = []
        for artist_idx, stats in self.artist_stats.items():
            if stats['n_users'] > 0:
                results.append({
                    'artist_idx': artist_idx,
                    'artist_name': self.mappings['idx_to_artist'][artist_idx],
                    **stats
                })

        results.sort(key=lambda x: x['n_users'], reverse=True)
        return results[:top_k]

    # ==================== RECOMMENDATION FUNCTIONALITY ====================

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        model: str = 'als',
        exclude_history: bool = True
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id: User ID string
            n: Number of recommendations
            model: Model to use ('als', 'ncf', 'hybrid')
            exclude_history: Whether to exclude already listened artists

        Returns:
            List of dictionaries with artist recommendations
        """
        # Find user
        user_info = self.search_users(user_id, method='exact')
        if not user_info:
            raise ValueError(f"User '{user_id}' not found")

        user_idx = user_info['user_idx']

        # Get recommendations based on model
        if model == 'als':
            recs = self.als_model.recommend_for_user(user_idx, n, exclude_history)
        elif model == 'ncf':
            if not self.ncf_model.trained:
                raise RuntimeError("NCF model not available. Use 'als' instead.")
            exclude_artists = list(self.user_history[user_idx]) if exclude_history else None
            recs = self.ncf_model.recommend_for_user(user_idx, n, exclude_artists)
        elif model == 'hybrid':
            recs = self._hybrid_recommend(user_idx, n, exclude_history)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Format results
        results = []
        for artist_idx, score in recs:
            results.append({
                'artist_idx': artist_idx,
                'artist_name': self.mappings['idx_to_artist'][artist_idx],
                'score': float(score),
                **self.artist_stats.get(artist_idx, {})
            })

        return results

    def _hybrid_recommend(
        self,
        user_idx: int,
        n: int,
        exclude_history: bool
    ) -> List[Tuple[int, float]]:
        """
        Hybrid recommendation combining ALS and NCF.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_history: Whether to exclude history

        Returns:
            List of (artist_idx, score) tuples
        """
        # Get recommendations from both models
        als_recs = self.als_model.recommend_for_user(user_idx, n * 2, exclude_history)

        if self.ncf_model.trained:
            exclude_artists = list(self.user_history[user_idx]) if exclude_history else None
            ncf_recs = self.ncf_model.recommend_for_user(user_idx, n * 2, exclude_artists)

            # Combine scores (weighted average)
            combined = {}
            for artist_idx, score in als_recs:
                combined[artist_idx] = 0.6 * score

            for artist_idx, score in ncf_recs:
                if artist_idx in combined:
                    combined[artist_idx] += 0.4 * score
                else:
                    combined[artist_idx] = 0.4 * score

            # Sort and get top-N
            sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:n]
        else:
            return als_recs[:n]

    def similar_artists(
        self,
        artist_name: str,
        n: int = 10,
        model: str = 'als'
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Find similar artists to a given artist.

        Args:
            artist_name: Artist name
            n: Number of similar artists
            model: Model to use ('als')

        Returns:
            List of dictionaries with similar artists
        """
        # Search for artist
        search_results = self.search_artists(artist_name, top_k=1, method='fuzzy')
        if not search_results:
            raise ValueError(f"Artist '{artist_name}' not found")

        artist_idx = search_results[0]['artist_idx']

        # Get similar artists
        if model == 'als':
            similar = self.als_model.similar_items(artist_idx, n)
        else:
            raise ValueError(f"Model '{model}' not supported for similarity")

        # Format results
        results = []
        for sim_artist_idx, score in similar:
            results.append({
                'artist_idx': sim_artist_idx,
                'artist_name': self.mappings['idx_to_artist'][sim_artist_idx],
                'similarity_score': float(score),
                **self.artist_stats.get(sim_artist_idx, {})
            })

        return results

    def get_user_profile(self, user_id: str, top_k: int = 10) -> Dict:
        """
        Get user profile with listening history.

        Args:
            user_id: User ID string
            top_k: Number of top artists to include

        Returns:
            Dictionary with user profile information
        """
        user_info = self.search_users(user_id, method='exact')
        if not user_info:
            raise ValueError(f"User '{user_id}' not found")

        user_idx = user_info['user_idx']

        # Get user's listening history
        user_data = self.train_df[self.train_df['user_idx'] == user_idx]
        user_data = user_data.sort_values('play_count', ascending=False)

        top_artists = []
        for _, row in user_data.head(top_k).iterrows():
            artist_idx = int(row['artist_idx'])
            top_artists.append({
                'artist_name': self.mappings['idx_to_artist'][artist_idx],
                'play_count': int(row['play_count']),
                'artist_idx': artist_idx
            })

        profile = {
            'user_id': user_id,
            'user_idx': user_idx,
            'total_interactions': len(user_data),
            'unique_artists': len(user_data),
            'total_plays': int(user_data['play_count'].sum()),
            'top_artists': top_artists
        }

        return profile

    def get_artist_profile(self, artist_name: str) -> Dict:
        """
        Get artist profile with statistics.

        Args:
            artist_name: Artist name

        Returns:
            Dictionary with artist profile information
        """
        search_results = self.search_artists(artist_name, top_k=1, method='fuzzy')
        if not search_results:
            raise ValueError(f"Artist '{artist_name}' not found")

        artist_info = search_results[0]
        return artist_info
