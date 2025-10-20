"""
Neural Collaborative Filtering (NCF) Model.
Deep learning-based recommendation using PyTorch.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from scipy import sparse
from config.config import Config
from tqdm import tqdm


class NCFDataset(Dataset):
    """PyTorch Dataset for NCF training."""

    def __init__(self, interactions: pd.DataFrame, n_artists: int, negative_sampling: bool = True):
        """
        Initialize NCF dataset.

        Args:
            interactions: DataFrame with user_idx, artist_idx, play_count_normalized
            n_artists: Total number of artists (for negative sampling)
            negative_sampling: Whether to generate negative samples
        """
        self.interactions = interactions
        self.n_artists = n_artists
        self.negative_sampling = negative_sampling

        # Create set of positive interactions for efficient lookup
        self.positive_set = set(
            zip(interactions['user_idx'].values, interactions['artist_idx'].values)
        )

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_idx = int(row['user_idx'])
        artist_idx = int(row['artist_idx'])
        rating = float(row['play_count_normalized'])

        # Normalize rating to [0, 1]
        rating = min(max(rating / 10.0, 0), 1)

        return user_idx, artist_idx, rating


class NCFModel(nn.Module):
    """Neural Collaborative Filtering model architecture."""

    def __init__(
        self,
        n_users: int,
        n_artists: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize NCF model.

        Args:
            n_users: Number of users
            n_artists: Number of artists
            embedding_dim: Dimension of embedding vectors
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(NCFModel, self).__init__()

        self.n_users = n_users
        self.n_artists = n_artists

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.artist_embedding = nn.Embedding(n_artists, embedding_dim)

        # MLP layers
        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.artist_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_idx, artist_idx):
        """
        Forward pass.

        Args:
            user_idx: Tensor of user indices
            artist_idx: Tensor of artist indices

        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_emb = self.user_embedding(user_idx)
        artist_emb = self.artist_embedding(artist_idx)

        # Concatenate embeddings
        x = torch.cat([user_emb, artist_emb], dim=-1)

        # Pass through MLP
        output = self.mlp(x)

        return output.squeeze()


class NCFRecommender:
    """NCF-based recommender system."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize NCF recommender.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_users = None
        self.n_artists = None
        self.trained = False

    def fit(
        self,
        train_df: pd.DataFrame,
        n_users: int,
        n_artists: int,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[float]]:
        """
        Train the NCF model.

        Args:
            train_df: Training DataFrame with user_idx, artist_idx, play_count_normalized
            n_users: Total number of users
            n_artists: Total number of artists
            val_df: Optional validation DataFrame

        Returns:
            Dictionary with training history
        """
        print("Training NCF model...")
        print(f"Device: {self.device}")
        print(f"Users: {n_users}, Artists: {n_artists}")

        self.n_users = n_users
        self.n_artists = n_artists

        # Create model
        self.model = NCFModel(
            n_users=n_users,
            n_artists=n_artists,
            embedding_dim=self.config.NCF_EMBEDDING_DIM,
            hidden_layers=self.config.NCF_HIDDEN_LAYERS,
            dropout=self.config.NCF_DROPOUT
        ).to(self.device)

        # Create dataset and dataloader
        train_dataset = NCFDataset(train_df, n_artists)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.NCF_BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.NCF_LEARNING_RATE)
        criterion = nn.MSELoss()

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.NCF_EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NCF_EPOCHS}")
            for user_idx, artist_idx, rating in pbar:
                user_idx = user_idx.to(self.device)
                artist_idx = artist_idx.to(self.device)
                rating = rating.to(self.device).float()

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(user_idx, artist_idx)
                loss = criterion(predictions, rating)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)

            # Validation
            if val_df is not None:
                val_loss = self._evaluate(val_df, criterion)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        self.trained = True
        print("NCF model training completed!")

        return history

    def _evaluate(self, df: pd.DataFrame, criterion) -> float:
        """Evaluate model on a dataset."""
        self.model.eval()
        dataset = NCFDataset(df, self.n_artists)
        loader = DataLoader(dataset, batch_size=self.config.NCF_BATCH_SIZE, shuffle=False)

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for user_idx, artist_idx, rating in loader:
                user_idx = user_idx.to(self.device)
                artist_idx = artist_idx.to(self.device)
                rating = rating.to(self.device).float()

                predictions = self.model(user_idx, artist_idx)
                loss = criterion(predictions, rating)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def predict(self, user_idx: int, artist_indices: List[int]) -> np.ndarray:
        """
        Predict ratings for user-artist pairs.

        Args:
            user_idx: User index
            artist_indices: List of artist indices

        Returns:
            Array of predicted ratings
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        self.model.eval()

        user_tensor = torch.tensor([user_idx] * len(artist_indices), dtype=torch.long).to(self.device)
        artist_tensor = torch.tensor(artist_indices, dtype=torch.long).to(self.device)

        with torch.no_grad():
            predictions = self.model(user_tensor, artist_tensor)

        return predictions.cpu().numpy()

    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 10,
        exclude_artists: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_idx: User index
            n: Number of recommendations
            exclude_artists: List of artist indices to exclude

        Returns:
            List of (artist_idx, score) tuples
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Get all artist indices
        all_artists = list(range(self.n_artists))

        # Exclude already interacted artists
        if exclude_artists:
            all_artists = [a for a in all_artists if a not in exclude_artists]

        # Predict ratings for all artists
        scores = self.predict(user_idx, all_artists)

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n]
        recommendations = [(all_artists[i], float(scores[i])) for i in top_indices]

        return recommendations

    def save_model(self, file_path: Optional[str] = None) -> None:
        """
        Save the trained model to disk.

        Args:
            file_path: Path to save model (uses config default if None)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        file_path = file_path or self.config.NCF_MODEL_FILE

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'n_users': self.n_users,
            'n_artists': self.n_artists,
            'config': {
                'embedding_dim': self.config.NCF_EMBEDDING_DIM,
                'hidden_layers': self.config.NCF_HIDDEN_LAYERS,
                'dropout': self.config.NCF_DROPOUT
            }
        }

        torch.save(checkpoint, file_path)
        print(f"Model saved to: {file_path}")

    def load_model(self, file_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to load model from (uses config default if None)
        """
        file_path = file_path or self.config.NCF_MODEL_FILE

        checkpoint = torch.load(file_path, map_location=self.device)

        self.n_users = checkpoint['n_users']
        self.n_artists = checkpoint['n_artists']

        config = checkpoint['config']
        self.model = NCFModel(
            n_users=self.n_users,
            n_artists=self.n_artists,
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout=config['dropout']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True

        print(f"Model loaded from: {file_path}")
