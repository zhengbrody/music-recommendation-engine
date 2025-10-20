"""Complete setup script: generate data, preprocess, and train models."""
import sys
from pathlib import Path

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR
from src.utils.data_generator import generate_sample_data, save_data
from src.preprocessing.data_loader import MusicDataLoader


def complete_setup():
    """Run complete setup process."""
    print("="*60)
    print("MUSIC RECOMMENDATION ENGINE - COMPLETE SETUP")
    print("="*60)

    # Step 1: Generate sample data
    print("\nStep 1: Generating sample data...")
    print("-"*60)
    interactions_df, users_df, songs_df = generate_sample_data(
        n_users=1000,
        n_songs=500,
        n_interactions=10000,
        seed=42
    )
    save_data(interactions_df, users_df, songs_df, RAW_DATA_DIR)

    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    print("-"*60)
    loader = MusicDataLoader(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    interactions, users, songs = loader.load_raw_data()
    processed_data = loader.preprocess_data(interactions, users, songs)
    loader.save_processed_data(processed_data)

    # Step 3: Train models
    print("\nStep 3: Training models...")
    print("-"*60)
    print("Please run: python train_models.py")
    print("\nOr import and run:")
    print("  from train_models import train_all_models")
    print("  train_all_models()")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train models: python train_models.py")
    print("2. Start API: python api/app.py")
    print("3. Start Dashboard: python dashboard/app.py")
    print("\nData generated:")
    print(f"  - {len(users_df)} users")
    print(f"  - {len(songs_df)} songs")
    print(f"  - {len(interactions_df)} interactions")


if __name__ == "__main__":
    complete_setup()

    # Ask if user wants to train models now
    print("\n" + "="*60)
    response = input("\nDo you want to train models now? (y/n): ")
    if response.lower() == 'y':
        from train_models import train_all_models
        train_all_models()
    else:
        print("\nYou can train models later by running: python train_models.py")
