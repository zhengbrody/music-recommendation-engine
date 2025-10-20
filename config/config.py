"""Configuration settings for the music recommendation engine."""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODEL_DIR = BASE_DIR / "models_saved"
MODEL_DIR.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5001))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"

# Dashboard Configuration
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8050))
DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "True").lower() == "true"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "False").lower() == "true"

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'music_rec.db'}"
)

# Spotify API Configuration (optional)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Model Configuration
N_FACTORS = 50  # Embedding dimensions
N_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Recommendation Configuration
TOP_N_RECOMMENDATIONS = 10
MIN_RATING = 1
MAX_RATING = 5
