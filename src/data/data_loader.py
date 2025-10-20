"""
Data loader for music recommendation system.
Loads raw data from Last.fm dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from config.config import Config


class DataLoader:
    """Load and parse Last.fm music listening data."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataLoader.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw Last.fm dataset from TSV file.

        Args:
            file_path: Path to TSV file (uses config default if None)

        Returns:
            DataFrame with columns: user_id, artist_mbid, artist_name, play_count

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if file_path is None:
            file_path = self.config.RAW_DATA_FILE

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please download the Last.fm dataset first."
            )

        print(f"Loading data from: {file_path}")

        # Load TSV file with proper column names
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['user_id', 'artist_mbid', 'artist_name', 'play_count'],
            encoding='utf-8',
            na_values=['', 'NA', 'null']
        )

        print(f"Loaded {len(df)} interactions for {df['user_id'].nunique()} users "
              f"and {df['artist_name'].nunique()} artists")

        return df

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Load preprocessed training and test data.

        Returns:
            Tuple of (train_df, test_df, mappings)

        Raises:
            FileNotFoundError: If processed data files don't exist
        """
        train_path = self.config.TRAIN_DATA_FILE
        test_path = self.config.TEST_DATA_FILE
        mappings_path = self.config.MAPPINGS_FILE

        if not all(os.path.exists(p) for p in [train_path, test_path, mappings_path]):
            raise FileNotFoundError(
                "Processed data not found. Please run preprocessing first."
            )

        print("Loading processed data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        import pickle
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)

        print(f"Train set: {len(train_df)} interactions")
        print(f"Test set: {len(test_df)} interactions")

        return train_df, test_df, mappings

    def get_dataset_stats(self, df: pd.DataFrame) -> dict:
        """
        Calculate statistics about the dataset.

        Args:
            df: DataFrame with user interactions

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_interactions': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_artists': df['artist_name'].nunique(),
            'avg_plays_per_user': df.groupby('user_id')['play_count'].sum().mean(),
            'avg_plays_per_artist': df.groupby('artist_name')['play_count'].sum().mean(),
            'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['artist_name'].nunique())),
            'min_plays': df['play_count'].min(),
            'max_plays': df['play_count'].max(),
            'median_plays': df['play_count'].median()
        }

        return stats
