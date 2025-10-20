"""Tests for recommendation models."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.collaborative_filtering import ImplicitALSModel, SurpriseSVDModel
from src.models.deep_learning import DeepRecommender


@pytest.fixture
def sample_data():
    """Create sample interaction data for testing."""
    np.random.seed(42)

    n_users = 100
    n_songs = 50
    n_interactions = 500

    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'song_id': np.random.randint(0, n_songs, n_interactions),
        'rating': np.random.randint(1, 6, n_interactions)
    })

    # Remove duplicates
    interactions = interactions.drop_duplicates(subset=['user_id', 'song_id'])

    # Create mappings
    user_id_map = {uid: idx for idx, uid in enumerate(sorted(interactions['user_id'].unique()))}
    song_id_map = {sid: idx for idx, sid in enumerate(sorted(interactions['song_id'].unique()))}

    interactions['user_idx'] = interactions['user_id'].map(user_id_map)
    interactions['song_idx'] = interactions['song_id'].map(song_id_map)

    return {
        'interactions': interactions,
        'user_id_map': user_id_map,
        'song_id_map': song_id_map,
        'n_users': len(user_id_map),
        'n_songs': len(song_id_map)
    }


def test_als_model(sample_data):
    """Test Implicit ALS model."""
    model = ImplicitALSModel(factors=10, iterations=5)
    model.prepare_data(
        sample_data['interactions'],
        sample_data['user_id_map'],
        sample_data['song_id_map']
    )
    model.train()

    # Test recommendations
    test_user = list(sample_data['user_id_map'].keys())[0]
    recs = model.recommend(test_user, n=5)

    assert len(recs) <= 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
    assert all(isinstance(r[0], (int, np.integer)) and isinstance(r[1], (float, np.floating)) for r in recs)


def test_svd_model(sample_data):
    """Test Surprise SVD model."""
    model = SurpriseSVDModel(n_factors=10, n_epochs=5)
    model.prepare_data(sample_data['interactions'])
    model.train()

    # Test recommendations
    test_user = list(sample_data['user_id_map'].keys())[0]
    recs = model.recommend(test_user, n=5, interactions_df=sample_data['interactions'])

    assert len(recs) <= 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)


def test_deep_learning_model(sample_data):
    """Test Deep Learning model."""
    model = DeepRecommender(
        n_users=sample_data['n_users'],
        n_songs=sample_data['n_songs'],
        n_factors=10,
        hidden_layers=[32, 16]
    )
    model.user_id_map = sample_data['user_id_map']
    model.song_id_map = sample_data['song_id_map']

    # Train for just 2 epochs for testing
    history = model.train(sample_data['interactions'], epochs=2, batch_size=32)

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2

    # Test recommendations
    test_user = list(sample_data['user_id_map'].keys())[0]
    recs = model.recommend(test_user, n=5, interactions_df=sample_data['interactions'])

    assert len(recs) <= 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
