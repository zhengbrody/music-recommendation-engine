"""
Data preprocessing for music recommendation system.
Handles data cleaning, filtering, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from config.config import Config


class DataPreprocessor:
    """Preprocess and clean music listening data."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by removing missing values and duplicates.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        initial_len = len(df)

        # Remove rows with missing user_id or artist_name
        df = df.dropna(subset=['user_id', 'artist_name'])

        # Remove rows with zero or negative play counts
        df = df[df['play_count'] > 0]

        # Remove duplicate interactions (keep the one with max play_count)
        df = df.groupby(['user_id', 'artist_name'], as_index=False).agg({
            'play_count': 'sum',
            'artist_mbid': 'first'
        })

        # Remove whitespace from artist names
        df['artist_name'] = df['artist_name'].str.strip()

        print(f"Removed {initial_len - len(df)} invalid/duplicate rows")
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to remove cold-start users and artists.

        Args:
            df: Cleaned DataFrame

        Returns:
            Filtered DataFrame with minimum interaction requirements
        """
        print("Filtering data...")
        initial_users = df['user_id'].nunique()
        initial_artists = df['artist_name'].nunique()

        # Iteratively filter until stable
        prev_len = 0
        iteration = 0
        while len(df) != prev_len:
            prev_len = len(df)
            iteration += 1

            # Filter users with minimum interactions
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.config.MIN_USER_INTERACTIONS].index
            df = df[df['user_id'].isin(valid_users)]

            # Filter artists with minimum interactions
            artist_counts = df['artist_name'].value_counts()
            valid_artists = artist_counts[artist_counts >= self.config.MIN_ARTIST_INTERACTIONS].index
            df = df[df['artist_name'].isin(valid_artists)]

            if iteration > 10:  # Safety check
                break

        final_users = df['user_id'].nunique()
        final_artists = df['artist_name'].nunique()

        print(f"Filtered: {initial_users - final_users} users, "
              f"{initial_artists - final_artists} artists")
        print(f"Remaining: {final_users} users, {final_artists} artists")

        return df

    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Create integer mappings for users and artists.

        Args:
            df: Filtered DataFrame

        Returns:
            Tuple of (DataFrame with integer IDs, mappings dictionary)
        """
        print("Creating ID mappings...")

        # Create user mappings
        unique_users = sorted(df['user_id'].unique())
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}

        # Create artist mappings
        unique_artists = sorted(df['artist_name'].unique())
        artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        idx_to_artist = {idx: artist for artist, idx in artist_to_idx.items()}

        # Apply mappings
        df['user_idx'] = df['user_id'].map(user_to_idx)
        df['artist_idx'] = df['artist_name'].map(artist_to_idx)

        # Store mappings
        mappings = {
            'user_to_idx': user_to_idx,
            'idx_to_user': idx_to_user,
            'artist_to_idx': artist_to_idx,
            'idx_to_artist': idx_to_artist,
            'n_users': len(unique_users),
            'n_artists': len(unique_artists)
        }

        print(f"Created mappings: {mappings['n_users']} users, {mappings['n_artists']} artists")

        return df, mappings

    def normalize_play_counts(self, df: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Normalize play counts to reduce skewness.

        Args:
            df: DataFrame with play_count column
            method: Normalization method ('log', 'minmax', 'standard')

        Returns:
            DataFrame with normalized play_count
        """
        if method == 'log':
            # Log transformation (add 1 to avoid log(0))
            df['play_count_normalized'] = np.log1p(df['play_count'])
        elif method == 'minmax':
            # Min-max normalization
            min_val = df['play_count'].min()
            max_val = df['play_count'].max()
            df['play_count_normalized'] = (df['play_count'] - min_val) / (max_val - min_val)
        elif method == 'standard':
            # Standardization (z-score)
            mean_val = df['play_count'].mean()
            std_val = df['play_count'].std()
            df['play_count_normalized'] = (df['play_count'] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return df

    def preprocess_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run full preprocessing pipeline.

        Args:
            df: Raw DataFrame

        Returns:
            Tuple of (preprocessed DataFrame, mappings)
        """
        # Clean data
        df = self.clean_data(df)

        # Filter data
        df = self.filter_data(df)

        # Normalize play counts
        df = self.normalize_play_counts(df, method='log')

        # Create mappings
        df, mappings = self.create_mappings(df)

        return df, mappings
