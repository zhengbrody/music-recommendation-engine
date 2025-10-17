import os
from dotenv import load_dotenv

# load.env file 
load_dotenv()

class Config:
    
    # ============ Path Configuration ============
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # ============ Data File ============
    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'usersha1-artmbid-artname-plays.tsv')
    TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    MAPPINGS_FILE = os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl')
    MATRIX_FILE = os.path.join(PROCESSED_DATA_DIR, 'interaction_matrix.npz')
    
    # ============ Model Dile ============
    ALS_MODEL_FILE = os.path.join(MODELS_DIR, 'als_model.pkl')
    NCF_MODEL_FILE = os.path.join(MODELS_DIR, 'ncf_model.pth')
    
    # ============ ALS Model Parameter ============
    ALS_FACTORS = 64                # Latent factor dimension
    ALS_REGULARIZATION = 0.01       # Regularization parameter
    ALS_ITERATIONS = 15             # Iteration count
    
    # ============ NCF Model Parameter ============
    NCF_EMBEDDING_DIM = 64          # Embedding Dimension
    NCF_HIDDEN_LAYERS = [128, 64, 32]  # MLP Hidden Layer
    NCF_DROPOUT = 0.2               # Dropout rare
    NCF_LEARNING_RATE = 0.001       # Learning Rate
    NCF_BATCH_SIZE = 256            # Batch Size
    NCF_EPOCHS = 10                 # Training Iterations
    NCF_NUM_NEGATIVES = 4           # Negative Sampling Ratio
    
    # ============ Data Process ============
    MIN_USER_INTERACTIONS = 5       # Minimum User Interactions
    MIN_ARTIST_INTERACTIONS = 10    # Minimum Artist Interactions
    TEST_SIZE = 0.2                 # Test Set Ratio
    RANDOM_STATE = 42               # Random State
    
    # ============ API Configuration ============
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    
    # ============ Redis Configuration ============
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = 0
    CACHE_EXPIRY = 3600  #Cache for 1 hour 
    
    # ============ PostgreSQL Configuration ============
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'music_rec_db')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'admin')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password123')
    
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # ============ Evaluation Parameters ============
    EVAL_K = 10  # Top-K Recommendations

class DevelopmentConfig(Config):
    """Development Environment Configuration"""
    FLASK_DEBUG = True

class ProductionConfig(Config):
    """Production Environment Configuration"""
    FLASK_DEBUG = False

# Configuration based on environment variables
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}