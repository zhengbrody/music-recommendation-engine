# Music Recommendation Engine - Quick Start Guide

## 🎉 Your System is Ready!

All components have been built and are running successfully!

## 📊 What's Running

### 1. **Flask API** - `http://localhost:5001`
Serving music recommendations via REST API

**Available Endpoints:**
```bash
# Health check
curl http://localhost:5001/health

# Get recommendations (ALS model)
curl "http://localhost:5001/recommend/0?model=als&n=10"

# Get recommendations (SVD model)
curl "http://localhost:5001/recommend/0?model=svd&n=10"

# Get recommendations (Deep Learning model)
curl "http://localhost:5001/recommend/0?model=deep&n=10"

# Get ensemble recommendations (combines all 3 models)
curl "http://localhost:5001/recommend/0?model=ensemble&n=10"

# Get user details
curl http://localhost:5001/user/0

# Get song details
curl http://localhost:5001/song/5

# Get system statistics
curl http://localhost:5001/stats
```

### 2. **Plotly Dash Dashboard** - `http://localhost:8050`
Interactive visualization dashboard with:
- Overview statistics (users, songs, interactions)
- Genre distribution charts
- User activity analysis
- Top songs ranking
- Song audio features by genre
- Rating distribution
- User demographics
- Interactive user recommendation explorer

## 📁 Project Components

### Data
- **1,000 users** with demographics (age, country, premium status)
- **500 songs** with audio features (tempo, energy, danceability, valence, acousticness)
- **9,627 interactions** (user ratings of songs)
- 10 genres: pop, rock, hip-hop, jazz, classical, electronic, country, r&b, indie, metal

### Models (All Trained and Ready)
1. **Implicit ALS** - Matrix factorization using alternating least squares
2. **Surprise SVD** - Singular value decomposition collaborative filtering
3. **Deep Learning** - Neural collaborative filtering with PyTorch
4. **Ensemble** - Combines predictions from all three models

## 🚀 How to Use

### Start the System (if not already running)

**Terminal 1 - API:**
```bash
cd /Users/zhengdong/music-recommendation-engine
PYTHONPATH=$(pwd) venv/bin/python api/app.py
```

**Terminal 2 - Dashboard:**
```bash
cd /Users/zhengdong/music-recommendation-engine
PYTHONPATH=$(pwd) venv/bin/python dashboard/app.py
```

### Test the API

```bash
# Get recommendations for user 0 using ensemble
curl -s "http://localhost:5001/recommend/0?model=ensemble&n=5" | python -m json.tool

# Get user 5's profile and listening history
curl -s http://localhost:5001/user/5 | python -m json.tool
```

### Explore the Dashboard

1. Open `http://localhost:8050` in your web browser
2. View statistics and visualizations
3. Scroll down to "User Recommendation Explorer"
4. Enter a user ID (0-999) to see their profile and listening history

## 📝 Example API Response

```json
{
    "user_id": 0,
    "model": "ensemble",
    "count": 5,
    "recommendations": [
        {
            "song_id": 356,
            "title": "Song_356",
            "artist": "Artist_146",
            "genre": "indie",
            "tempo": 118.66,
            "energy": 0.56,
            "danceability": 0.98,
            "recommendation_score": 4.54
        },
        ...
    ]
}
```

## 🧪 Run Tests

```bash
cd /Users/zhengdong/music-recommendation-engine
venv/bin/pytest tests/ -v
```

## 🔄 Retrain Models

If you want to regenerate data and retrain models:

```bash
# Generate new sample data
PYTHONPATH=$(pwd) venv/bin/python src/utils/data_generator.py

# Preprocess the data
PYTHONPATH=$(pwd) venv/bin/python src/preprocessing/data_loader.py

# Train all models
PYTHONPATH=$(pwd) venv/bin/python train_models.py
```

## 📚 Project Structure

```
music-recommendation-engine/
├── api/                          # Flask REST API
│   └── app.py                    # API endpoints
├── dashboard/                    # Plotly Dash dashboard
│   └── app.py                    # Dashboard with visualizations
├── src/
│   ├── models/                   # ML models
│   │   ├── collaborative_filtering.py  # ALS & SVD models
│   │   └── deep_learning.py            # PyTorch neural CF
│   ├── preprocessing/            # Data processing
│   │   └── data_loader.py
│   └── utils/
│       └── data_generator.py     # Sample data generation
├── data/
│   ├── raw/                      # Original data (CSV files)
│   └── processed/                # Preprocessed data
├── models_saved/                 # Trained models
│   ├── als_model.pkl
│   ├── svd_model.pkl
│   └── deep_model.pt
├── config/                       # Configuration
│   └── config.py
├── tests/                        # Unit tests
├── train_models.py              # Model training script
├── setup_and_run.py             # Complete setup script
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Claude Code documentation
└── README_USAGE.md              # Detailed usage guide
```

## 🎯 Next Steps

1. **Explore Different Models**: Try comparing recommendations from `als`, `svd`, `deep`, and `ensemble`
2. **Tune Hyperparameters**: Edit `config/config.py` to adjust model parameters
3. **Add More Data**: Modify `src/utils/data_generator.py` to create more users/songs
4. **Integrate Spotify API**: Add your credentials to use real Spotify data
5. **Deploy**: Use gunicorn for production deployment of the API

## 💡 Tips

- User IDs range from 0-999
- Song IDs range from 0-499
- The ensemble model typically gives the best recommendations
- Check the dashboard for insights into user preferences and song popularity
- All models are cached in memory for fast recommendations

## 🛠️ Troubleshooting

**API not responding?**
- Check if port 5001 is available
- View API logs in the terminal where you started it

**Dashboard not loading?**
- Check if port 8050 is available
- Ensure the preprocessed data exists in `data/processed/`

**Model training errors?**
- Ensure all dependencies are installed: `venv/bin/pip install -r requirements.txt`
- Check that numpy version is < 2.0 for compatibility

## 📞 Support

See `CLAUDE.md` for development documentation and `README_USAGE.md` for detailed usage instructions.

---

**Enjoy your music recommendation engine!** 🎵
