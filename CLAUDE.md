# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A complete Spotify-style music recommendation system combining collaborative filtering and deep learning. The system includes:
- 3 trained ML models (ALS, SVD, Neural CF) + ensemble
- Flask REST API (port 5001)
- Plotly Dash visualization dashboard (port 8050)
- Synthetic dataset: 1,000 users, 500 songs, 9,627 interactions

## Quick Start

**First-time setup:**
```bash
# Install dependencies (numpy<2 required for scikit-surprise compatibility)
venv/bin/pip install -r requirements.txt

# Generate data, preprocess, and train models
PYTHONPATH=$(pwd) venv/bin/python setup_and_run.py
# Then run: PYTHONPATH=$(pwd) venv/bin/python train_models.py
```

**Start services:**
```bash
# Option 1: Start both services
./start_all.sh

# Option 2: Start manually
PYTHONPATH=$(pwd) venv/bin/python api/app.py      # Terminal 1
PYTHONPATH=$(pwd) venv/bin/python dashboard/app.py # Terminal 2
```

**Access:**
- API: http://localhost:5001
- Dashboard: http://localhost:8050

## Development Commands

### Data Generation & Training
```bash
# Generate synthetic music data
PYTHONPATH=$(pwd) venv/bin/python src/utils/data_generator.py

# Preprocess data (creates user-item matrix, normalizes features, creates mappings)
PYTHONPATH=$(pwd) venv/bin/python src/preprocessing/data_loader.py

# Train all three models (ALS, SVD, Deep Learning)
PYTHONPATH=$(pwd) venv/bin/python train_models.py
```

### Running Services
```bash
# API server (Flask with CORS)
PYTHONPATH=$(pwd) venv/bin/python api/app.py

# Dashboard (Plotly Dash with Bootstrap)
PYTHONPATH=$(pwd) venv/bin/python dashboard/app.py
```

### Testing
```bash
# Run all tests
venv/bin/pytest tests/ -v

# Run specific test
venv/bin/pytest tests/test_models.py -v

# Test API endpoints
curl http://localhost:5001/health
curl "http://localhost:5001/recommend/0?model=ensemble&n=5"
```

## Architecture

### Three-Model Hybrid System

**1. Implicit ALS (collaborative_filtering.py)**
- Matrix factorization using Alternating Least Squares
- Optimized for implicit feedback (play counts, ratings)
- Uses scipy sparse matrices for efficiency
- Model saved as `models_saved/als_model.pkl`

**2. Surprise SVD (collaborative_filtering.py)**
- Traditional collaborative filtering with SVD
- Handles explicit ratings (1-5 scale)
- Predicts user-song affinity scores
- Model saved as `models_saved/svd_model.pkl`

**3. Neural Collaborative Filtering (deep_learning.py)**
- PyTorch neural network with user/song embeddings
- MLP architecture: [128, 64, 32] hidden layers with dropout
- Learns non-linear interaction patterns
- Model saved as `models_saved/deep_model.pt`
- **Important**: Uses `weights_only=False` in torch.load() due to numpy compatibility

**4. Ensemble Method (api/app.py)**
- Combines predictions from all three models
- Averages recommendation scores
- Typically provides best overall performance

### Data Flow

```
Raw Data (CSV) → Data Generator → data/raw/
                                     ↓
                                 Preprocessing → data/processed/
                                     ↓
                                 Model Training → models_saved/
                                     ↓
                           API loads models (on startup)
                                     ↓
                           Serves recommendations via REST
```

### API Design (api/app.py)

**Key Pattern:** Models are loaded into memory on startup for fast inference

**Endpoints:**
- `GET /` - API documentation
- `GET /health` - Health check (returns models_loaded count)
- `GET /models` - List available models
- `GET /recommend/<user_id>?model=<als|svd|deep|ensemble>&n=<num>` - Get recommendations
- `GET /song/<song_id>` - Song metadata
- `GET /user/<user_id>` - User profile + listening history
- `GET /stats` - System statistics (users, songs, interactions, genres)

**Important Implementation Details:**
- ALS model's `recommend()` method does NOT accept `interactions_df` parameter
- SVD and Deep models DO accept `interactions_df` for filtering
- Ensemble mode handles different model interfaces correctly

### Dashboard Architecture (dashboard/app.py)

**Built with Plotly Dash + Bootstrap components**

Interactive visualizations:
- Overview cards (metrics summary)
- Genre distribution (pie chart)
- User activity patterns (histograms)
- Top songs ranking (horizontal bars)
- Audio features by genre (box plots)
- User recommendation explorer (callback-based search)

**Callback Pattern:** Uses `@app.callback` decorators for interactivity
- `update_feature_chart()` - Updates audio feature visualization based on dropdown
- `show_user_info()` - Fetches and displays user profile on button click

### Configuration (config/config.py)

**All settings in one place:**
- API port: 5001 (changed from 5000 due to AirPlay conflict)
- Dashboard port: 8050
- Model hyperparameters (N_FACTORS=50, N_EPOCHS=10)
- Database: SQLite by default (PostgreSQL optional)
- Redis: Disabled by default (optional caching)

**Environment Variables:** Use `.env` file (see `.env.example`)

### Data Schema

**users.csv:** user_id, age, country, premium
**songs.csv:** song_id, title, artist, genre, duration_ms, tempo, energy, danceability, valence, acousticness
**interactions.csv:** user_id, song_id, rating, timestamp

**Processed data:**
- `user_item_matrix.csv` - Sparse matrix representation
- `interactions_processed.csv` - With user_idx, song_idx mappings
- `songs_features.csv` - Normalized audio features
- `mappings.pkl` - ID-to-index mappings and StandardScaler

### Model Training Workflow

1. **Data Generation** (src/utils/data_generator.py)
   - Creates realistic user preferences based on genre affinities
   - 70% chance users listen to preferred genres
   - Higher ratings for preferred genres

2. **Preprocessing** (src/preprocessing/data_loader.py)
   - Creates sparse user-item matrix
   - Normalizes song features using StandardScaler
   - Generates user_id → user_idx and song_id → song_idx mappings

3. **Training** (train_models.py)
   - Trains all three models sequentially
   - Saves models to `models_saved/` directory
   - Prints sample recommendations for validation

### Important Dependencies

**NumPy version constraint:** Must use `numpy<2` for scikit-surprise compatibility
- scikit-surprise uses Cython extensions compiled against NumPy 1.x
- NumPy 2.x breaks binary compatibility
- If you see import errors, downgrade: `pip install "numpy<2"`

**PyTorch loading:** Deep learning model uses `weights_only=False` to handle numpy scalars in saved state

## Common Development Tasks

### Adding a New Model

1. Implement in `src/models/` with standardized interface:
   - `train(interactions_df, **kwargs)` method
   - `recommend(user_id, n=10)` method
   - `save(filepath)` and `load(filepath)` methods

2. Add to `train_models.py` training pipeline

3. Update `api/app.py` to load and serve the model

4. Add tests in `tests/test_models.py`

### Modifying Hyperparameters

Edit `config/config.py`:
- `N_FACTORS` - Embedding dimensions (default: 50)
- `N_EPOCHS` - Training iterations (default: 10)
- `BATCH_SIZE` - Deep learning batch size (default: 256)
- `LEARNING_RATE` - Deep learning learning rate (default: 0.001)

Then retrain: `PYTHONPATH=$(pwd) venv/bin/python train_models.py`

### Using Real Spotify Data

Currently uses synthetic data. To integrate real Spotify data:

1. Get Spotify API credentials from https://developer.spotify.com
2. Add to `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_id
   SPOTIFY_CLIENT_SECRET=your_secret
   ```
3. Use `spotipy` library (already in requirements.txt)
4. Replace `data_generator.py` with Spotify API calls

## Troubleshooting

**Port conflicts:**
- API uses port 5001 (not 5000, to avoid macOS AirPlay)
- Dashboard uses port 8050

**Import errors:**
- Always run with `PYTHONPATH=$(pwd)` to ensure proper module imports
- The project structure requires parent directory in path

**Model loading errors:**
- Ensure all three models exist in `models_saved/`
- Run `train_models.py` if models are missing
- Check NumPy version: must be < 2.0

**scikit-learn version warnings:**
- StandardScaler may show version mismatch warnings
- These are non-critical (data was pickled with older scikit-learn)
- Models still work correctly

## Project Structure Notes

- `notebooks/` exists but is empty (reserved for exploratory analysis)
- `setup.py` exists but is empty (project uses direct imports)
- Redis and PostgreSQL are optional (system works with SQLite)
- Locust load testing file mentioned in original CLAUDE.md was not implemented
