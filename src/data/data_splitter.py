"""
Data splitting for train/test sets.
Implements temporal and random splitting strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scipy import sparse
from config.config import Config
import pickle
import os


class DataSplitter:
    """Split data into training and test sets."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataSplitter.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

    def random_split(
        self,
        df: pd.DataFrame,
        test_ratio: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly split data into train and test sets.

        Args:
            df: DataFrame with user interactions
            test_ratio: Fraction for test set (uses config default if None)
            random_state: Random seed (uses config default if None)

        Returns:
            Tuple of (train_df, test_df)
        """
        test_ratio = test_ratio or self.config.TEST_RATIO
        random_state = random_state or self.config.RANDOM_STATE

        print(f"Performing random split ({int((1-test_ratio)*100)}% train, "
              f"{int(test_ratio*100)}% test)...")

        # Shuffle and split
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_ratio))

        train_df = df[:split_idx].copy()
        test_df = df[split_idx:].copy()

        print(f"Train set: {len(train_df)} interactions")
        print(f"Test set: {len(test_df)} interactions")

        return train_df, test_df

    def temporal_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        test_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on timestamps (if available).

        Args:
            df: DataFrame with user interactions
            timestamp_col: Column name containing timestamps
            test_ratio: Fraction for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        test_ratio = test_ratio or self.config.TEST_RATIO

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        print(f"Performing temporal split on column '{timestamp_col}'...")

        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_ratio))

        train_df = df[:split_idx].copy()
        test_df = df[split_idx:].copy()

        return train_df, test_df

    def user_based_split(
        self,
        df: pd.DataFrame,
        test_ratio: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by holding out interactions per user.

        This ensures each user appears in both train and test sets.

        Args:
            df: DataFrame with user interactions
            test_ratio: Fraction of each user's interactions for test
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        test_ratio = test_ratio or self.config.TEST_RATIO
        random_state = random_state or self.config.RANDOM_STATE

        print(f"Performing user-based split...")

        np.random.seed(random_state)
        train_list = []
        test_list = []

        for user_id in df['user_idx'].unique():
            user_data = df[df['user_idx'] == user_id].copy()

            if len(user_data) < 2:
                # If user has only 1 interaction, put it in train
                train_list.append(user_data)
                continue

            # Shuffle user's interactions
            user_data = user_data.sample(frac=1, random_state=random_state)

            # Split
            n_test = max(1, int(len(user_data) * test_ratio))
            test_list.append(user_data[:n_test])
            train_list.append(user_data[n_test:])

        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

        print(f"Train set: {len(train_df)} interactions ({train_df['user_idx'].nunique()} users)")
        print(f"Test set: {len(test_df)} interactions ({test_df['user_idx'].nunique()} users)")

        return train_df, test_df

    def create_interaction_matrix(
        self,
        df: pd.DataFrame,
        n_users: int,
        n_artists: int,
        value_col: str = 'play_count_normalized'
    ) -> sparse.csr_matrix:
        """
        Create sparse user-artist interaction matrix.

        Args:
            df: DataFrame with user_idx, artist_idx, and value columns
            n_users: Total number of users
            n_artists: Total number of artists
            value_col: Column to use for matrix values

        Returns:
            Sparse CSR matrix of shape (n_users, n_artists)
        """
        print(f"Creating interaction matrix ({n_users} x {n_artists})...")

        # Create sparse matrix
        matrix = sparse.csr_matrix(
            (df[value_col].values, (df['user_idx'].values, df['artist_idx'].values)),
            shape=(n_users, n_artists)
        )

        density = matrix.nnz / (n_users * n_artists)
        print(f"Matrix density: {density:.6f}")

        return matrix

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mappings: dict,
        train_matrix: Optional[sparse.csr_matrix] = None
    ) -> None:
        """
        Save processed data to disk.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            mappings: User/artist mappings
            train_matrix: Optional sparse interaction matrix
        """
        print("Saving processed data...")

        # Save DataFrames
        train_df.to_csv(self.config.TRAIN_DATA_FILE, index=False)
        test_df.to_csv(self.config.TEST_DATA_FILE, index=False)
        print(f"Saved train data to: {self.config.TRAIN_DATA_FILE}")
        print(f"Saved test data to: {self.config.TEST_DATA_FILE}")

        # Save mappings
        with open(self.config.MAPPINGS_FILE, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"Saved mappings to: {self.config.MAPPINGS_FILE}")

        # Save interaction matrix
        if train_matrix is not None:
            sparse.save_npz(self.config.INTERACTION_MATRIX_FILE, train_matrix)
            print(f"Saved interaction matrix to: {self.config.INTERACTION_MATRIX_FILE}")

        print("All data saved successfully!")
