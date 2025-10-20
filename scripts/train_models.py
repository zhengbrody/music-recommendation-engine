"""
Training script for music recommendation models.
Trains both ALS and NCF models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.data_splitter import DataSplitter
from src.models.als import ALSRecommender
from src.models.ncf import NCFRecommender
from config.config import Config
import argparse


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train music recommendation models')
    parser.add_argument('--model', type=str, default='both', choices=['als', 'ncf', 'both'],
                       help='Model to train')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing (use existing processed data)')
    args = parser.parse_args()

    config = Config()

    print("="*70)
    print("MUSIC RECOMMENDATION MODEL TRAINING")
    print("="*70)

    # Step 1: Load and preprocess data
    if not args.skip_preprocessing:
        print("\n[Step 1/4] Loading raw data...")
        loader = DataLoader(config)

        try:
            raw_df = loader.load_raw_data()
            print(f"✓ Loaded {len(raw_df)} raw interactions")
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("\nPlease download the Last.fm dataset first:")
            print("  python scripts/download_data.py")
            return

        print("\n[Step 2/4] Preprocessing data...")
        preprocessor = DataPreprocessor(config)
        processed_df, mappings = preprocessor.preprocess_pipeline(raw_df)
        print(f"✓ Processed data: {len(processed_df)} interactions")
        print(f"✓ Users: {mappings['n_users']}, Artists: {mappings['n_artists']}")

        print("\n[Step 3/4] Splitting data...")
        splitter = DataSplitter(config)
        train_df, test_df = splitter.user_based_split(processed_df)

        # Create interaction matrix for ALS
        train_matrix = splitter.create_interaction_matrix(
            train_df,
            mappings['n_users'],
            mappings['n_artists']
        )

        # Save processed data
        splitter.save_processed_data(train_df, test_df, mappings, train_matrix)
        print("✓ Processed data saved")
    else:
        print("\n[Step 1-3/4] Loading preprocessed data...")
        loader = DataLoader(config)
        train_df, test_df, mappings = loader.load_processed_data()

        from scipy import sparse
        train_matrix = sparse.load_npz(config.INTERACTION_MATRIX_FILE)
        print("✓ Preprocessed data loaded")

    # Step 4: Train models
    print("\n[Step 4/4] Training models...")

    if args.model in ['als', 'both']:
        print("\n" + "-"*70)
        print("Training ALS Model")
        print("-"*70)
        als_model = ALSRecommender(config)
        als_model.fit(train_matrix)
        als_model.save_model()
        print("✓ ALS model trained and saved")

    if args.model in ['ncf', 'both']:
        print("\n" + "-"*70)
        print("Training NCF Model")
        print("-"*70)
        ncf_model = NCFRecommender(config)
        ncf_model.fit(
            train_df,
            mappings['n_users'],
            mappings['n_artists']
        )
        ncf_model.save_model()
        print("✓ NCF model trained and saved")

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Evaluate models: python scripts/evaluate_models.py")
    print("  2. Start API server: python src/api/app.py")
    print("  3. Try search demo: python scripts/search_demo.py")


if __name__ == '__main__':
    main()
