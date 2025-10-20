"""
Interactive demo showcasing search functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.recommender import MusicRecommender
from config.config import Config
import argparse


def print_artists(results, title="Results"):
    """Pretty print artist search results."""
    print(f"\n{title}")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['artist_name']}")
        if 'match_score' in result:
            print(f"   Match Score: {result['match_score']:.3f}")
        if 'total_plays' in result:
            print(f"   Total Plays: {result['total_plays']:,} | Unique Listeners: {result['n_users']:,}")
        if 'score' in result:
            print(f"   Recommendation Score: {result['score']:.4f}")
        if 'similarity_score' in result:
            print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print()


def search_demo(recommender):
    """Interactive search demo."""
    print("\n" + "="*80)
    print("MUSIC SEARCH DEMO")
    print("="*80)

    while True:
        print("\nSearch Options:")
        print("  1. Search artists by name")
        print("  2. Get popular artists")
        print("  3. Get trending artists")
        print("  4. Find similar artists")
        print("  5. Search users")
        print("  6. Get user profile")
        print("  7. Get recommendations for user")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        try:
            if choice == '1':
                query = input("Enter artist name: ").strip()
                method = input("Search method (fuzzy/exact/contains) [fuzzy]: ").strip() or 'fuzzy'
                results = recommender.search_artists(query, top_k=10, method=method)
                print_artists(results, f"Search Results for '{query}' ({method})")

            elif choice == '2':
                results = recommender.search_by_popularity(top_k=10)
                print_artists(results, "Top 10 Most Popular Artists")

            elif choice == '3':
                results = recommender.search_by_user_count(top_k=10)
                print_artists(results, "Top 10 Trending Artists (Most Listeners)")

            elif choice == '4':
                artist_name = input("Enter artist name: ").strip()
                results = recommender.similar_artists(artist_name, n=10)
                print_artists(results, f"Artists Similar to '{artist_name}'")

            elif choice == '5':
                user_id = input("Enter user ID: ").strip()
                result = recommender.search_users(user_id)
                if result:
                    print("\nUser Found:")
                    print(f"  User ID: {result['user_id']}")
                    print(f"  User Index: {result['user_idx']}")
                    print(f"  Interactions: {result['n_interactions']}")
                else:
                    print("\n✗ User not found")

            elif choice == '6':
                user_id = input("Enter user ID: ").strip()
                profile = recommender.get_user_profile(user_id, top_k=10)
                print("\nUser Profile:")
                print(f"  User ID: {profile['user_id']}")
                print(f"  Total Interactions: {profile['total_interactions']}")
                print(f"  Unique Artists: {profile['unique_artists']}")
                print(f"  Total Plays: {profile['total_plays']:,}")
                print("\nTop Artists:")
                for i, artist in enumerate(profile['top_artists'], 1):
                    print(f"    {i}. {artist['artist_name']} - {artist['play_count']} plays")

            elif choice == '7':
                user_id = input("Enter user ID: ").strip()
                model = input("Model (als/ncf/hybrid) [als]: ").strip() or 'als'
                recommendations = recommender.recommend(user_id, n=10, model=model)
                print_artists(recommendations, f"Recommendations for User '{user_id}' ({model} model)")

            elif choice == '0':
                print("\nExiting demo. Goodbye!")
                break

            else:
                print("\n✗ Invalid choice. Please try again.")

        except ValueError as e:
            print(f"\n✗ Error: {e}")
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")


def quick_demo(recommender):
    """Non-interactive quick demo."""
    print("\n" + "="*80)
    print("QUICK SEARCH DEMO")
    print("="*80)

    # Demo 1: Search artists
    print("\n[Demo 1] Searching for 'Beatles'...")
    results = recommender.search_artists("beatles", top_k=5, method='fuzzy')
    print_artists(results, "Search Results for 'Beatles'")

    # Demo 2: Popular artists
    print("\n[Demo 2] Top 5 Popular Artists...")
    results = recommender.search_by_popularity(top_k=5)
    print_artists(results, "Top 5 Most Popular Artists")

    # Demo 3: Similar artists
    print("\n[Demo 3] Artists Similar to 'Radiohead'...")
    try:
        results = recommender.similar_artists("radiohead", n=5)
        print_artists(results, "Artists Similar to 'Radiohead'")
    except ValueError:
        print("Radiohead not found in dataset, trying 'coldplay'...")
        try:
            results = recommender.similar_artists("coldplay", n=5)
            print_artists(results, "Artists Similar to 'Coldplay'")
        except ValueError:
            print("Could not find artist for similarity demo")

    print("\n" + "="*80)
    print("For interactive demo, run without --quick flag")
    print("="*80)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Search and recommendation demo')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick non-interactive demo')
    args = parser.parse_args()

    config = Config()

    print("="*80)
    print("LOADING MUSIC RECOMMENDER SYSTEM")
    print("="*80)

    try:
        recommender = MusicRecommender(config)
        recommender.load_data_and_models()
        print("\n✓ System loaded successfully!")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease train models first:")
        print("  python scripts/train_models.py")
        return
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return

    # Run demo
    if args.quick:
        quick_demo(recommender)
    else:
        search_demo(recommender)


if __name__ == '__main__':
    main()
