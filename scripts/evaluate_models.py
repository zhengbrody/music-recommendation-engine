"""
Evaluation script for trained models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader
from src.models.als import ALSRecommender
from src.models.ncf import NCFRecommender
from src.evaluation.evaluator import ModelEvaluator
from config.config import Config
import argparse


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', type=str, default='both', choices=['als', 'ncf', 'both'],
                       help='Model to evaluate')
    parser.add_argument('--n-users', type=int, default=None,
                       help='Number of test users to evaluate (None = all)')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-K for evaluation metrics')
    args = parser.parse_args()

    config = Config()

    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Load data
    print("\n[1/3] Loading data...")
    loader = DataLoader(config)
    train_df, test_df, mappings = loader.load_processed_data()
    print(f"✓ Data loaded: {mappings['n_users']} users, {mappings['n_artists']} artists")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Evaluate ALS
    if args.model in ['als', 'both']:
        print("\n[2/3] Evaluating ALS Model...")
        print("-"*70)
        try:
            als_model = ALSRecommender(config)
            als_model.load_model()

            als_metrics = evaluator.evaluate_model(
                model=als_model,
                test_df=test_df,
                train_df=train_df,
                n_users=args.n_users,
                n_artists=mappings['n_artists'],
                k=args.k
            )

            print("\nALS Model Results:")
            evaluator.print_metrics(als_metrics)
        except FileNotFoundError:
            print("✗ ALS model not found. Train it first: python scripts/train_models.py --model als")

    # Evaluate NCF
    if args.model in ['ncf', 'both']:
        print("\n[3/3] Evaluating NCF Model...")
        print("-"*70)
        try:
            ncf_model = NCFRecommender(config)
            ncf_model.load_model()

            ncf_metrics = evaluator.evaluate_model(
                model=ncf_model,
                test_df=test_df,
                train_df=train_df,
                n_users=args.n_users,
                n_artists=mappings['n_artists'],
                k=args.k
            )

            print("\nNCF Model Results:")
            evaluator.print_metrics(ncf_metrics)
        except FileNotFoundError:
            print("✗ NCF model not found. Train it first: python scripts/train_models.py --model ncf")

    # Compare models
    if args.model == 'both':
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        models = {}
        try:
            als_model = ALSRecommender(config)
            als_model.load_model()
            models['ALS'] = als_model
        except:
            pass

        try:
            ncf_model = NCFRecommender(config)
            ncf_model.load_model()
            models['NCF'] = ncf_model
        except:
            pass

        if len(models) >= 2:
            comparison_df = evaluator.compare_models(
                models=models,
                test_df=test_df,
                train_df=train_df,
                n_users=args.n_users,
                n_artists=mappings['n_artists'],
                k=args.k
            )
            print("\n", comparison_df)

    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
