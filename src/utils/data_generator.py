"""Generate synthetic music interaction data for testing."""
import numpy as np
import pandas as pd
from pathlib import Path
import json


def generate_sample_data(n_users=1000, n_songs=500, n_interactions=10000, seed=42):
    """Generate synthetic music listening data.

    Args:
        n_users: Number of users
        n_songs: Number of songs
        n_interactions: Number of user-song interactions
        seed: Random seed for reproducibility

    Returns:
        tuple: (interactions_df, users_df, songs_df)
    """
    np.random.seed(seed)

    # Generate song metadata
    genres = ['pop', 'rock', 'hip-hop', 'jazz', 'classical', 'electronic',
              'country', 'r&b', 'indie', 'metal']
    artists = [f'Artist_{i}' for i in range(200)]

    songs_data = {
        'song_id': range(n_songs),
        'title': [f'Song_{i}' for i in range(n_songs)],
        'artist': np.random.choice(artists, n_songs),
        'genre': np.random.choice(genres, n_songs),
        'duration_ms': np.random.randint(120000, 300000, n_songs),  # 2-5 minutes
        'tempo': np.random.uniform(60, 180, n_songs),
        'energy': np.random.uniform(0, 1, n_songs),
        'danceability': np.random.uniform(0, 1, n_songs),
        'valence': np.random.uniform(0, 1, n_songs),  # Musical positivity
        'acousticness': np.random.uniform(0, 1, n_songs),
    }
    songs_df = pd.DataFrame(songs_data)

    # Generate user metadata
    users_data = {
        'user_id': range(n_users),
        'age': np.random.randint(13, 70, n_users),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR'], n_users),
        'premium': np.random.choice([True, False], n_users, p=[0.3, 0.7]),
    }
    users_df = pd.DataFrame(users_data)

    # Generate interactions with some patterns
    # Create genre preferences for users
    user_genre_prefs = {}
    for user_id in range(n_users):
        # Each user has 1-3 preferred genres
        n_prefs = np.random.randint(1, 4)
        user_genre_prefs[user_id] = np.random.choice(genres, n_prefs, replace=False)

    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)

        # 70% chance of listening to preferred genre
        if np.random.random() < 0.7 and user_id in user_genre_prefs:
            preferred_genres = user_genre_prefs[user_id]
            song_candidates = songs_df[songs_df['genre'].isin(preferred_genres)]
            if len(song_candidates) > 0:
                song_id = song_candidates.sample(1)['song_id'].values[0]
            else:
                song_id = np.random.randint(0, n_songs)
        else:
            song_id = np.random.randint(0, n_songs)

        # Generate rating (implicit feedback: play count / listen time)
        # More weight towards higher ratings for preferred genres
        song_genre = songs_df.loc[song_id, 'genre']
        if user_id in user_genre_prefs and song_genre in user_genre_prefs[user_id]:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
        else:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.3, 0.2, 0.1])

        interactions.append({
            'user_id': user_id,
            'song_id': song_id,
            'rating': rating,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        })

    interactions_df = pd.DataFrame(interactions)

    # Remove duplicates, keep highest rating
    interactions_df = interactions_df.sort_values('rating', ascending=False)
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'song_id'], keep='first')

    return interactions_df, users_df, songs_df


def save_data(interactions_df, users_df, songs_df, data_dir):
    """Save generated data to CSV files.

    Args:
        interactions_df: User-song interactions
        users_df: User metadata
        songs_df: Song metadata
        data_dir: Directory to save data
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    interactions_df.to_csv(data_dir / 'interactions.csv', index=False)
    users_df.to_csv(data_dir / 'users.csv', index=False)
    songs_df.to_csv(data_dir / 'songs.csv', index=False)

    print(f"Data saved to {data_dir}")
    print(f"  - Interactions: {len(interactions_df)}")
    print(f"  - Users: {len(users_df)}")
    print(f"  - Songs: {len(songs_df)}")


if __name__ == "__main__":
    from config.config import RAW_DATA_DIR

    print("Generating sample music data...")
    interactions_df, users_df, songs_df = generate_sample_data(
        n_users=1000,
        n_songs=500,
        n_interactions=10000
    )

    save_data(interactions_df, users_df, songs_df, RAW_DATA_DIR)
    print("\nData generation complete!")
