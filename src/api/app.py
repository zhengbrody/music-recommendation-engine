"""
Flask API for Music Recommendation System with Search.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.recommender import MusicRecommender
from src.utils.logger import setup_logger
from src.utils.cache import CacheManager
from src.utils.database import DatabaseManager
from config.config import Config


def create_app(config: Optional[Config] = None) -> Flask:
    """
    Create and configure Flask application.

    Args:
        config: Configuration object (uses default if None)

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)

    # Configuration
    cfg = config or Config()
    app.config['JSON_SORT_KEYS'] = False

    # Setup logger
    logger = setup_logger()

    # Initialize components
    recommender = MusicRecommender(cfg)
    cache = CacheManager(cfg)
    database = DatabaseManager(cfg)

    # Load data and models
    try:
        recommender.load_data_and_models()
        logger.info("Recommender system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        raise

    # ==================== HEALTH CHECK ====================

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'service': 'music-recommendation-api',
            'cache_enabled': cache.enabled,
            'database_enabled': database.enabled
        })

    # ==================== SEARCH ENDPOINTS ====================

    @app.route('/api/search/artists', methods=['GET'])
    def search_artists():
        """
        Search for artists by name.

        Query Parameters:
            - q: Search query (required)
            - top_k: Number of results (default: 10)
            - method: Search method - 'fuzzy', 'exact', 'contains' (default: 'fuzzy')
        """
        query = request.args.get('q', '')
        top_k = int(request.args.get('top_k', 10))
        method = request.args.get('method', 'fuzzy')

        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400

        # Check cache
        cache_key = f"search:artists:{query}:{top_k}:{method}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for artist search: {query}")
            return jsonify(cached)

        try:
            results = recommender.search_artists(query, top_k, method)

            # Log to database
            database.log_search(query, 'artist', len(results))

            response = {
                'query': query,
                'method': method,
                'n_results': len(results),
                'results': results
            }

            # Cache results
            cache.set(cache_key, response)

            return jsonify(response)

        except Exception as e:
            logger.error(f"Artist search error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/search/users', methods=['GET'])
    def search_users():
        """
        Search for users by ID.

        Query Parameters:
            - user_id: User ID (required)
            - method: Search method - 'exact', 'contains' (default: 'exact')
        """
        user_id = request.args.get('user_id', '')
        method = request.args.get('method', 'exact')

        if not user_id:
            return jsonify({'error': 'Query parameter "user_id" is required'}), 400

        try:
            result = recommender.search_users(user_id, method)

            if result:
                return jsonify({'found': True, 'user': result})
            else:
                return jsonify({'found': False, 'message': 'User not found'}), 404

        except Exception as e:
            logger.error(f"User search error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/search/popular', methods=['GET'])
    def search_popular():
        """
        Get most popular artists.

        Query Parameters:
            - top_k: Number of results (default: 10)
        """
        top_k = int(request.args.get('top_k', 10))

        # Check cache
        cache_key = f"search:popular:{top_k}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)

        try:
            results = recommender.search_by_popularity(top_k)

            response = {
                'n_results': len(results),
                'results': results
            }

            # Cache results
            cache.set(cache_key, response)

            return jsonify(response)

        except Exception as e:
            logger.error(f"Popular search error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/search/trending', methods=['GET'])
    def search_trending():
        """
        Get artists with most unique listeners.

        Query Parameters:
            - top_k: Number of results (default: 10)
        """
        top_k = int(request.args.get('top_k', 10))

        # Check cache
        cache_key = f"search:trending:{top_k}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)

        try:
            results = recommender.search_by_user_count(top_k)

            response = {
                'n_results': len(results),
                'results': results
            }

            # Cache results
            cache.set(cache_key, response)

            return jsonify(response)

        except Exception as e:
            logger.error(f"Trending search error: {e}")
            return jsonify({'error': str(e)}), 500

    # ==================== RECOMMENDATION ENDPOINTS ====================

    @app.route('/api/recommend', methods=['GET'])
    def recommend():
        """
        Get personalized recommendations for a user.

        Query Parameters:
            - user_id: User ID (required)
            - n: Number of recommendations (default: 10)
            - model: Model to use - 'als', 'ncf', 'hybrid' (default: 'als')
            - exclude_history: Whether to exclude listened artists (default: true)
        """
        user_id = request.args.get('user_id', '')
        n = int(request.args.get('n', 10))
        model = request.args.get('model', 'als')
        exclude_history = request.args.get('exclude_history', 'true').lower() == 'true'

        if not user_id:
            return jsonify({'error': 'Query parameter "user_id" is required'}), 400

        # Check cache
        cache_key = f"recommend:{user_id}:{n}:{model}:{exclude_history}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for recommendations: {user_id}")
            return jsonify(cached)

        try:
            recommendations = recommender.recommend(user_id, n, model, exclude_history)

            response = {
                'user_id': user_id,
                'model': model,
                'n_recommendations': len(recommendations),
                'recommendations': recommendations
            }

            # Log to database
            for rec in recommendations:
                database.log_recommendation(user_id, rec['artist_idx'], rec['score'], model)

            # Cache results
            cache.set(cache_key, response)

            return jsonify(response)

        except ValueError as e:
            logger.warning(f"Recommendation error: {e}")
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/similar/artists', methods=['GET'])
    def similar_artists():
        """
        Find similar artists.

        Query Parameters:
            - artist_name: Artist name (required)
            - n: Number of similar artists (default: 10)
            - model: Model to use (default: 'als')
        """
        artist_name = request.args.get('artist_name', '')
        n = int(request.args.get('n', 10))
        model = request.args.get('model', 'als')

        if not artist_name:
            return jsonify({'error': 'Query parameter "artist_name" is required'}), 400

        # Check cache
        cache_key = f"similar:artists:{artist_name}:{n}:{model}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)

        try:
            similar = recommender.similar_artists(artist_name, n, model)

            response = {
                'artist_name': artist_name,
                'model': model,
                'n_similar': len(similar),
                'similar_artists': similar
            }

            # Cache results
            cache.set(cache_key, response)

            return jsonify(response)

        except ValueError as e:
            logger.warning(f"Similar artists error: {e}")
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"Similar artists error: {e}")
            return jsonify({'error': str(e)}), 500

    # ==================== PROFILE ENDPOINTS ====================

    @app.route('/api/profile/user', methods=['GET'])
    def user_profile():
        """
        Get user profile with listening history.

        Query Parameters:
            - user_id: User ID (required)
            - top_k: Number of top artists to include (default: 10)
        """
        user_id = request.args.get('user_id', '')
        top_k = int(request.args.get('top_k', 10))

        if not user_id:
            return jsonify({'error': 'Query parameter "user_id" is required'}), 400

        # Check cache
        cache_key = f"profile:user:{user_id}:{top_k}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)

        try:
            profile = recommender.get_user_profile(user_id, top_k)

            # Cache results
            cache.set(cache_key, profile)

            return jsonify(profile)

        except ValueError as e:
            logger.warning(f"User profile error: {e}")
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"User profile error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/profile/artist', methods=['GET'])
    def artist_profile():
        """
        Get artist profile with statistics.

        Query Parameters:
            - artist_name: Artist name (required)
        """
        artist_name = request.args.get('artist_name', '')

        if not artist_name:
            return jsonify({'error': 'Query parameter "artist_name" is required'}), 400

        # Check cache
        cache_key = f"profile:artist:{artist_name}"
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)

        try:
            profile = recommender.get_artist_profile(artist_name)

            # Cache results
            cache.set(cache_key, profile)

            return jsonify(profile)

        except ValueError as e:
            logger.warning(f"Artist profile error: {e}")
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            logger.error(f"Artist profile error: {e}")
            return jsonify({'error': str(e)}), 500

    # ==================== STATS ENDPOINT ====================

    @app.route('/api/stats', methods=['GET'])
    def stats():
        """Get system statistics."""
        try:
            stats_data = {
                'n_users': recommender.mappings['n_users'],
                'n_artists': recommender.mappings['n_artists'],
                'n_interactions': len(recommender.train_df),
                'sparsity': 1 - (len(recommender.train_df) /
                               (recommender.mappings['n_users'] * recommender.mappings['n_artists'])),
                'cache_enabled': cache.enabled,
                'database_enabled': database.enabled
            }

            return jsonify(stats_data)

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({'error': str(e)}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    config = Config()
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=True
    )
