# Music Recommendation Engine

A production-ready Spotify-style music recommendation system with collaborative filtering, deep learning, and comprehensive search functionality.

## Features

### Recommendation Algorithms
- **ALS (Alternating Least Squares)**: Matrix factorization-based collaborative filtering using the implicit library
- **NCF (Neural Collaborative Filtering)**: Deep learning-based recommendations using PyTorch
- **Hybrid Model**: Combines ALS and NCF for improved accuracy

### Search Functionality
- **Artist Search**: Fuzzy, exact, and substring matching for finding artists
- **User Search**: Locate users by ID
- **Popularity Search**: Discover most played artists
- **Trending Search**: Find artists with most unique listeners
- **Similar Artists**: Content-based similarity recommendations
- **User Profiles**: View listening history and preferences

### API Features
- RESTful Flask API with CORS support
- Redis caching for fast response times
- PostgreSQL database for logging and analytics
- Comprehensive error handling
- Automatic request/response logging

### Evaluation Metrics
- Precision@K, Recall@K, F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Hit Rate@K
- Coverage and Diversity metrics
- Novelty scoring

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/music-recommendation-engine.git
cd music-recommendation-engine

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings (optional)
```

### 2. Download Dataset

```bash
python scripts/download_data.py
```

This downloads the Last.fm 360K dataset (~200 MB) containing user listening history.

### 3. Train Models

```bash
# Train both ALS and NCF models
python scripts/train_models.py

# Or train specific model
python scripts/train_models.py --model als
python scripts/train_models.py --model ncf
```

Training typically takes:
- ALS: 2-5 minutes
- NCF: 10-30 minutes (depending on GPU availability)

### 4. Test Search Functionality

```bash
# Interactive search demo
python scripts/search_demo.py

# Quick demo
python scripts/search_demo.py --quick
```

### 5. Start API Server

```bash
python src/api/app.py
```

API will be available at `http://localhost:5000`

## API Documentation

### Search Endpoints

#### Search Artists
```bash
GET /api/search/artists?q=beatles&top_k=10&method=fuzzy
```

Parameters:
- `q` (required): Search query
- `top_k` (optional, default=10): Number of results
- `method` (optional, default='fuzzy'): Search method ('fuzzy', 'exact', 'contains')

Response:
```json
{
  "query": "beatles",
  "method": "fuzzy",
  "n_results": 5,
  "results": [
    {
      "artist_idx": 123,
      "artist_name": "The Beatles",
      "match_score": 0.95,
      "total_plays": 150000,
      "n_users": 5000
    }
  ]
}
```

#### Search Popular Artists
```bash
GET /api/search/popular?top_k=10
```

Returns most played artists by total play count.

#### Search Trending Artists
```bash
GET /api/search/trending?top_k=10
```

Returns artists with most unique listeners.

#### Search Users
```bash
GET /api/search/users?user_id=abc123&method=exact
```

Find user by ID with exact or partial matching.

### Recommendation Endpoints

#### Get Recommendations
```bash
GET /api/recommend?user_id=abc123&n=10&model=als&exclude_history=true
```

Parameters:
- `user_id` (required): User ID
- `n` (optional, default=10): Number of recommendations
- `model` (optional, default='als'): Model to use ('als', 'ncf', 'hybrid')
- `exclude_history` (optional, default=true): Exclude already listened artists

Response:
```json
{
  "user_id": "abc123",
  "model": "als",
  "n_recommendations": 10,
  "recommendations": [
    {
      "artist_idx": 456,
      "artist_name": "Radiohead",
      "score": 0.856,
      "total_plays": 80000,
      "n_users": 3000
    }
  ]
}
```

#### Find Similar Artists
```bash
GET /api/similar/artists?artist_name=radiohead&n=10
```

Returns artists similar to the specified artist.

### Profile Endpoints

#### Get User Profile
```bash
GET /api/profile/user?user_id=abc123&top_k=10
```

Returns user listening history and top artists.

#### Get Artist Profile
```bash
GET /api/profile/artist?artist_name=coldplay
```

Returns artist statistics and popularity metrics.

### System Endpoints

#### Health Check
```bash
GET /health
```

Returns system status and component availability.

#### Statistics
```bash
GET /api/stats
```

Returns dataset and system statistics.

## Architecture

```
music-recommendation-engine/
├── config/
│   └── config.py              # Configuration management
├── data/
│   ├── raw/                   # Raw Last.fm dataset
│   └── processed/             # Preprocessed train/test splits
├── models/                    # Saved model files
├── src/
│   ├── data/                  # Data loading and preprocessing
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── data_splitter.py
│   ├── models/                # Recommendation models
│   │   ├── als.py
│   │   ├── ncf.py
│   │   └── recommender.py    # Main recommender with search
│   ├── evaluation/            # Metrics and evaluation
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── api/                   # Flask REST API
│   │   └── app.py
│   └── utils/                 # Utilities
│       ├── logger.py
│       ├── cache.py
│       └── database.py
├── scripts/                   # Training and demo scripts
│   ├── download_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── search_demo.py
└── tests/                     # Unit tests

```

## Data Processing Pipeline

1. **Data Loading**: Load Last.fm user-artist-plays dataset
2. **Cleaning**: Remove missing values, duplicates, invalid entries
3. **Filtering**: Apply minimum interaction thresholds for users and artists
4. **Mapping**: Create integer indices for users and artists
5. **Normalization**: Log-transform play counts
6. **Splitting**: User-based train/test split (80/20)
7. **Matrix Creation**: Generate sparse interaction matrix

## Model Training

### ALS (Alternating Least Squares)

Matrix factorization approach that learns user and item embeddings:

```python
# Default configuration
factors = 64
regularization = 0.01
iterations = 15
```

### NCF (Neural Collaborative Filtering)

Deep learning model with user/item embeddings and MLP layers:

```python
# Default configuration
embedding_dim = 64
hidden_layers = [128, 64, 32]
dropout = 0.2
learning_rate = 0.001
batch_size = 256
epochs = 10
```

## Evaluation

Evaluate trained models:

```bash
# Evaluate both models
python scripts/evaluate_models.py

# Evaluate specific model
python scripts/evaluate_models.py --model als --k 10 --n-users 1000
```

Metrics computed:
- Precision@K, Recall@K, F1@K
- NDCG@K
- MAP (Mean Average Precision)
- Hit Rate@K
- Coverage (catalog coverage)
- Diversity (recommendation diversity)
- Novelty (popularity bias)

## Search Functionality Details

The system provides comprehensive search capabilities:

### Artist Search Methods

1. **Fuzzy Search** (default): Uses sequence matching for typo-tolerant search
   - Example: "beatls" matches "The Beatles"

2. **Exact Search**: Requires exact match (case-insensitive)
   - Example: "radiohead" matches only "Radiohead"

3. **Contains Search**: Substring matching
   - Example: "cold" matches "Coldplay", "Cold War Kids"

### Search Features

- **Relevance Scoring**: Results ranked by match quality
- **Statistics**: Each result includes play counts, listener counts
- **Caching**: Fast repeated searches via Redis
- **Logging**: All searches logged to database for analytics

## Configuration

Edit `config/config.py` or use environment variables:

```python
# Model parameters
ALS_FACTORS = 64
NCF_EMBEDDING_DIM = 64
NCF_HIDDEN_LAYERS = [128, 64, 32]

# Data processing
MIN_USER_INTERACTIONS = 5
MIN_ARTIST_INTERACTIONS = 10
TEST_RATIO = 0.2

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000

# Cache
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CACHE_EXPIRE = 3600  # 1 hour
```

## Performance

Typical performance on Last.fm 360K dataset:

- **Dataset**: ~360K users, ~300K artists, ~17M interactions
- **Training Time**: ALS (5 min), NCF (30 min on GPU)
- **API Latency**: <50ms (cached), <200ms (uncached)
- **Memory**: ~2GB (ALS), ~4GB (NCF)
- **Metrics**: Precision@10 ~0.15, NDCG@10 ~0.25

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint
flake8 src/ scripts/ tests/
```

### Adding New Features

1. Models: Extend `src/models/`
2. Metrics: Add to `src/evaluation/metrics.py`
3. API endpoints: Update `src/api/app.py`
4. Search methods: Enhance `src/models/recommender.py`

## Troubleshooting

### Dataset Download Issues

If automatic download fails:
1. Manual download: http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz
2. Extract TSV file to `data/raw/usersha1-artmbid-artname-plays.tsv`

### Redis Connection Errors

Redis is optional. System works without caching if Redis is unavailable.

To install Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

### GPU Support for NCF

Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Citation

If you use this system in research, please cite:

```
Last.fm Dataset: http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- Last.fm for the dataset
- Implicit library for ALS implementation
- PyTorch for deep learning framework
- Flask for API framework
