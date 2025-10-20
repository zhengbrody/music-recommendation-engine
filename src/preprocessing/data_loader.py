"""Data loading and preprocessing functions."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


class MusicDataLoader:
    """Load and preprocess music recommendation data."""

    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()

    def load_raw_data(self):
        """Load raw data files."""
        interactions = pd.read_csv(self.raw_data_dir / 'interactions.csv')
        users = pd.read_csv(self.raw_data_dir / 'users.csv')
        songs = pd.read_csv(self.raw_data_dir / 'songs.csv')

        # Convert timestamp
        if 'timestamp' in interactions.columns:
            interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])

        return interactions, users, songs

    def preprocess_data(self, interactions, users, songs):
        """Preprocess data for model training."""
        # Create user-item matrix
        user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='song_id',
            values='rating',
            fill_value=0
        )

        # Normalize song features
        feature_cols = ['tempo', 'energy', 'danceability', 'valence', 'acousticness']
        songs_features = songs[['song_id'] + feature_cols].copy()
        songs_features[feature_cols] = self.scaler.fit_transform(songs_features[feature_cols])

        # Create mappings
        user_id_map = {user_id: idx for idx, user_id in enumerate(sorted(interactions['user_id'].unique()))}
        song_id_map = {song_id: idx for idx, song_id in enumerate(sorted(interactions['song_id'].unique()))}

        # Map IDs to indices
        interactions_mapped = interactions.copy()
        interactions_mapped['user_idx'] = interactions_mapped['user_id'].map(user_id_map)
        interactions_mapped['song_idx'] = interactions_mapped['song_id'].map(song_id_map)

        return {
            'interactions': interactions_mapped,
            'user_item_matrix': user_item_matrix,
            'users': users,
            'songs': songs,
            'songs_features': songs_features,
            'user_id_map': user_id_map,
            'song_id_map': song_id_map,
            'scaler': self.scaler
        }

    def save_processed_data(self, data_dict):
        """Save processed data."""
        # Save DataFrames
        data_dict['interactions'].to_csv(
            self.processed_data_dir / 'interactions_processed.csv', index=False
        )
        data_dict['user_item_matrix'].to_csv(
            self.processed_data_dir / 'user_item_matrix.csv'
        )
        data_dict['songs_features'].to_csv(
            self.processed_data_dir / 'songs_features.csv', index=False
        )

        # Save mappings and scaler
        with open(self.processed_data_dir / 'mappings.pkl', 'wb') as f:
            pickle.dump({
                'user_id_map': data_dict['user_id_map'],
                'song_id_map': data_dict['song_id_map'],
                'scaler': data_dict['scaler']
            }, f)

        print(f"Processed data saved to {self.processed_data_dir}")

    def load_processed_data(self):
        """Load previously processed data."""
        interactions = pd.read_csv(self.processed_data_dir / 'interactions_processed.csv')
        user_item_matrix = pd.read_csv(self.processed_data_dir / 'user_item_matrix.csv', index_col=0)
        songs_features = pd.read_csv(self.processed_data_dir / 'songs_features.csv')

        with open(self.processed_data_dir / 'mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)

        return {
            'interactions': interactions,
            'user_item_matrix': user_item_matrix,
            'songs_features': songs_features,
            **mappings
        }


if __name__ == "__main__":
    from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    loader = MusicDataLoader(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    print("Loading raw data...")
    interactions, users, songs = loader.load_raw_data()

    print("Preprocessing data...")
    processed_data = loader.preprocess_data(interactions, users, songs)

    print("Saving processed data...")
    loader.save_processed_data(processed_data)

    print("\nPreprocessing complete!")
    print(f"  - Interactions: {len(processed_data['interactions'])}")
    print(f"  - Users: {len(processed_data['user_id_map'])}")
    print(f"  - Songs: {len(processed_data['song_id_map'])}")
