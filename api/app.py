"""Flask API for serving music recommendations."""
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    API_HOST, API_PORT, API_DEBUG, MODEL_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from src.models.collaborative_filtering import ImplicitALSModel, SurpriseSVDModel
from src.models.deep_learning import DeepRecommender

app = Flask(__name__)
CORS(app)

# Global variables for models and data
models = {}
songs_df = None
users_df = None
interactions_df = None


def load_models_and_data():
    """Load trained models and data."""
    global models, songs_df, users_df, interactions_df

    print("Loading data...")
    # Load data
    songs_df = pd.read_csv(RAW_DATA_DIR / 'songs.csv')
    users_df = pd.read_csv(RAW_DATA_DIR / 'users.csv')
    interactions_df = pd.read_csv(PROCESSED_DATA_DIR / 'interactions_processed.csv')

    print("Loading models...")
    # Load ALS model
    try:
        als_model = ImplicitALSModel()
        als_model.load(MODEL_DIR / 'als_model.pkl')
        models['als'] = als_model
        print("  ✓ ALS model loaded")
    except Exception as e:
        print(f"  ✗ ALS model not found: {e}")

    # Load SVD model
    try:
        svd_model = SurpriseSVDModel()
        svd_model.load(MODEL_DIR / 'svd_model.pkl')
        models['svd'] = svd_model
        print("  ✓ SVD model loaded")
    except Exception as e:
        print(f"  ✗ SVD model not found: {e}")

    # Load Deep Learning model
    try:
        # Load mappings first
        with open(PROCESSED_DATA_DIR / 'mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)

        n_users = len(mappings['user_id_map'])
        n_songs = len(mappings['song_id_map'])

        dl_model = DeepRecommender(n_users, n_songs)
        dl_model.load(MODEL_DIR / 'deep_model.pt')
        dl_model.user_id_map = mappings['user_id_map']
        dl_model.song_id_map = mappings['song_id_map']
        models['deep'] = dl_model
        print("  ✓ Deep Learning model loaded")
    except Exception as e:
        print(f"  ✗ Deep Learning model not found: {e}")

    print(f"Loaded {len(models)} models")


@app.route('/')
def index():
    """API index."""
    return jsonify({
        'message': 'Music Recommendation API',
        'version': '1.0',
        'endpoints': {
            '/health': 'Health check',
            '/models': 'List available models',
            '/recommend/<user_id>': 'Get recommendations for a user',
            '/song/<song_id>': 'Get song details',
            '/user/<user_id>': 'Get user details',
            '/stats': 'Get system statistics'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'data_loaded': songs_df is not None
    })


@app.route('/models')
def list_models():
    """List available models."""
    return jsonify({
        'models': list(models.keys()),
        'count': len(models)
    })


@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    """Get recommendations for a user.

    Query parameters:
        - model: Model to use (als, svd, deep, ensemble) [default: als]
        - n: Number of recommendations [default: 10]
    """
    model_name = request.args.get('model', 'als')
    n = int(request.args.get('n', 10))

    if model_name not in models and model_name != 'ensemble':
        return jsonify({'error': f'Model {model_name} not available'}), 404

    # Get recommendations
    try:
        if model_name == 'ensemble':
            # Ensemble: combine all models
            all_recs = {}
            for name, model in models.items():
                if name == 'als':
                    recs = model.recommend(user_id, n=n)
                else:
                    recs = model.recommend(user_id, n=n, interactions_df=interactions_df)
                for song_id, score in recs:
                    if song_id not in all_recs:
                        all_recs[song_id] = []
                    all_recs[song_id].append(score)

            # Average scores
            ensemble_recs = [
                (song_id, sum(scores) / len(scores))
                for song_id, scores in all_recs.items()
            ]
            ensemble_recs.sort(key=lambda x: x[1], reverse=True)
            recommendations = ensemble_recs[:n]
        else:
            model = models[model_name]
            if model_name == 'als':
                recommendations = model.recommend(user_id, n=n)
            else:
                recommendations = model.recommend(user_id, n=n, interactions_df=interactions_df)

        # Enrich with song details
        enriched_recs = []
        for song_id, score in recommendations:
            song_info = songs_df[songs_df['song_id'] == song_id].iloc[0].to_dict()
            song_info['recommendation_score'] = float(score)
            enriched_recs.append(song_info)

        return jsonify({
            'user_id': user_id,
            'model': model_name,
            'recommendations': enriched_recs,
            'count': len(enriched_recs)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/song/<int:song_id>')
def get_song(song_id):
    """Get song details."""
    song = songs_df[songs_df['song_id'] == song_id]

    if len(song) == 0:
        return jsonify({'error': 'Song not found'}), 404

    return jsonify(song.iloc[0].to_dict())


@app.route('/user/<int:user_id>')
def get_user(user_id):
    """Get user details and listening history."""
    user = users_df[users_df['user_id'] == user_id]

    if len(user) == 0:
        return jsonify({'error': 'User not found'}), 404

    # Get user's listening history
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    user_interactions = user_interactions.merge(songs_df, on='song_id')
    history = user_interactions[['song_id', 'title', 'artist', 'genre', 'rating']].to_dict('records')

    user_data = user.iloc[0].to_dict()
    user_data['listening_history'] = history
    user_data['total_interactions'] = len(history)

    return jsonify(user_data)


@app.route('/stats')
def get_stats():
    """Get system statistics."""
    return jsonify({
        'total_users': len(users_df),
        'total_songs': len(songs_df),
        'total_interactions': len(interactions_df),
        'models_available': list(models.keys()),
        'genres': songs_df['genre'].unique().tolist(),
        'genre_distribution': songs_df['genre'].value_counts().to_dict()
    })


if __name__ == '__main__':
    print("Starting Music Recommendation API...")
    load_models_and_data()
    print(f"\nAPI running on http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
