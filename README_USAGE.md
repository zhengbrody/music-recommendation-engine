# Music Recommendation Engine - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Data and Train Models

**Option A: Automatic Setup (Recommended)**
```bash
python setup_and_run.py
```

This will:
- Generate sample music data (1000 users, 500 songs, 10,000 interactions)
- Preprocess the data
- Prompt you to train models

**Option B: Manual Steps**
```bash
# Generate sample data
python src/utils/data_generator.py

# Preprocess data
python src/preprocessing/data_loader.py

# Train all models
python train_models.py
```

### 3. Run the API

```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

**API Endpoints:**
- `GET /` - API documentation
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /recommend/<user_id>?model=als&n=10` - Get recommendations
  - Models: `als`, `svd`, `deep`, `ensemble`
- `GET /song/<song_id>` - Get song details
- `GET /user/<user_id>` - Get user profile and history
- `GET /stats` - System statistics

**Example API Usage:**
```bash
# Get recommendations for user 0 using ALS model
curl http://localhost:5000/recommend/0?model=als&n=10

# Get ensemble recommendations
curl http://localhost:5000/recommend/0?model=ensemble&n=10

# Get user profile
curl http://localhost:5000/user/0

# Get song details
curl http://localhost:5000/song/5

# Get system stats
curl http://localhost:5000/stats
```

### 4. Run the Dashboard

```bash
python dashboard/app.py
```

The dashboard will be available at `http://localhost:8050`

**Dashboard Features:**
- Overview statistics (users, songs, interactions)
- Genre distribution charts
- User activity analysis
- Top songs ranking
- Song audio features by genre
- Rating distribution
- Interactive user explorer (search by user ID)

## Project Structure

```
music-recommendation-engine/
├── api/                    # Flask REST API
│   └── app.py
├── config/                 # Configuration
│   └── config.py
├── dashboard/              # Plotly Dash dashboard
│   └── app.py
├── data/
│   ├── raw/               # Generated data
│   └── processed/         # Preprocessed data
├── models_saved/          # Trained models
├── src/
│   ├── models/            # ML models
│   │   ├── collaborative_filtering.py  # ALS & SVD
│   │   └── deep_learning.py            # Neural CF
│   ├── preprocessing/     # Data processing
│   │   └── data_loader.py
│   └── utils/
│       └── data_generator.py
├── tests/                 # Unit tests
├── setup_and_run.py      # Complete setup script
├── train_models.py       # Train all models
└── requirements.txt
```

## Models

### 1. Implicit ALS (Alternating Least Squares)
- Fast matrix factorization
- Good for implicit feedback
- File: `models_saved/als_model.pkl`

### 2. Surprise SVD
- Traditional collaborative filtering
- Explicit rating predictions
- File: `models_saved/svd_model.pkl`

### 3. Deep Learning (Neural Collaborative Filtering)
- Neural network approach
- Learns complex patterns
- File: `models_saved/deep_model.pt`

### 4. Ensemble
- Combines all three models
- Averages predictions
- Available via API only

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## Configuration

Environment variables can be set in `.env` file (copy from `.env.example`):

```bash
# Copy example
cp .env.example .env

# Edit as needed
nano .env
```

## Troubleshooting

### PyTorch Installation Issues
If PyTorch fails to install, try:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Redis (Optional)
Redis is optional for caching. To enable:
1. Install Redis: `brew install redis` (macOS) or `apt-get install redis` (Linux)
2. Start Redis: `redis-server`
3. Set `REDIS_ENABLED=True` in `.env`

### PostgreSQL (Optional)
By default, uses SQLite. To use PostgreSQL:
1. Install PostgreSQL
2. Create database: `createdb music_rec`
3. Update `DATABASE_URL` in `.env`

## Next Steps

1. **Customize Data**: Modify `src/utils/data_generator.py` to change data generation
2. **Tune Models**: Adjust hyperparameters in `config/config.py`
3. **Add Features**: Extend models in `src/models/`
4. **Integrate Spotify**: Add your Spotify API credentials to use real data
5. **Deploy**: Use gunicorn for production API serving

## Example Workflow

```bash
# 1. Setup
python setup_and_run.py

# 2. Train models
python train_models.py

# 3. Terminal 1: Start API
python api/app.py

# 4. Terminal 2: Start Dashboard
python dashboard/app.py

# 5. Terminal 3: Test API
curl http://localhost:5000/recommend/0?model=ensemble&n=10

# 6. Open browser
# API: http://localhost:5000
# Dashboard: http://localhost:8050
```

Enjoy your music recommendation engine! 🎵
