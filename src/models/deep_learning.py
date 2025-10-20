"""Deep learning recommendation model using PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import pickle


class MusicInteractionDataset(Dataset):
    """Dataset for user-song interactions."""

    def __init__(self, interactions_df):
        self.users = torch.LongTensor(interactions_df['user_idx'].values)
        self.songs = torch.LongTensor(interactions_df['song_idx'].values)
        self.ratings = torch.FloatTensor(interactions_df['rating'].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.ratings[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model."""

    def __init__(self, n_users, n_songs, n_factors=50, hidden_layers=[128, 64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.song_embedding = nn.Embedding(n_songs, n_factors)

        # MLP layers
        layers = []
        input_size = n_factors * 2

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.song_embedding.weight, std=0.01)

    def forward(self, user_ids, song_ids):
        """Forward pass."""
        user_embedded = self.user_embedding(user_ids)
        song_embedded = self.song_embedding(song_ids)

        # Concatenate embeddings
        x = torch.cat([user_embedded, song_embedded], dim=-1)

        # Pass through MLP
        output = self.mlp(x)

        return output.squeeze()


class DeepRecommender:
    """Deep learning recommender system."""

    def __init__(self, n_users, n_songs, n_factors=50, hidden_layers=[128, 64, 32],
                 learning_rate=0.001, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuralCollaborativeFiltering(
            n_users, n_songs, n_factors, hidden_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.n_users = n_users
        self.n_songs = n_songs
        self.user_id_map = None
        self.song_id_map = None

    def train(self, interactions_df, epochs=10, batch_size=256, validation_split=0.2):
        """Train the model.

        Args:
            interactions_df: DataFrame with user_idx, song_idx, rating
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation

        Returns:
            Training history
        """
        # Split data
        train_size = int(len(interactions_df) * (1 - validation_split))
        train_df = interactions_df.iloc[:train_size]
        val_df = interactions_df.iloc[train_size:]

        # Create datasets
        train_dataset = MusicInteractionDataset(train_df)
        val_dataset = MusicInteractionDataset(val_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {'train_loss': [], 'val_loss': []}

        print("Training Deep Learning model...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for users, songs, ratings in train_loader:
                users = users.to(self.device)
                songs = songs.to(self.device)
                ratings = ratings.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(users, songs)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, songs, ratings in val_loader:
                    users = users.to(self.device)
                    songs = songs.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = self.model(users, songs)
                    loss = self.criterion(predictions, ratings)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("Training complete!")
        return history

    def recommend(self, user_id, n=10, interactions_df=None):
        """Get recommendations for a user.

        Args:
            user_id: Original user ID
            n: Number of recommendations
            interactions_df: DataFrame to filter already interacted items

        Returns:
            List of (song_id, score) tuples
        """
        if user_id not in self.user_id_map:
            return []

        user_idx = self.user_id_map[user_id]
        idx_to_song = {v: k for k, v in self.song_id_map.items()}

        # Get candidate songs
        if interactions_df is not None:
            user_songs = set(interactions_df[interactions_df['user_id'] == user_id]['song_id'])
            candidate_song_ids = [sid for sid in self.song_id_map.keys() if sid not in user_songs]
            candidate_indices = [self.song_id_map[sid] for sid in candidate_song_ids]
        else:
            candidate_indices = list(range(self.n_songs))
            candidate_song_ids = [idx_to_song[i] for i in candidate_indices]

        # Predict ratings
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(candidate_indices)).to(self.device)
            song_tensor = torch.LongTensor(candidate_indices).to(self.device)

            predictions = self.model(user_tensor, song_tensor).cpu().numpy()

        # Sort and return top N
        top_indices = np.argsort(predictions)[::-1][:n]
        recommendations = [
            (candidate_song_ids[i], float(predictions[i]))
            for i in top_indices
        ]

        return recommendations

    def save(self, filepath):
        """Save model to disk."""
        filepath = Path(filepath)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_users': self.n_users,
            'n_songs': self.n_songs,
            'user_id_map': self.user_id_map,
            'song_id_map': self.song_id_map
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.n_users = checkpoint['n_users']
        self.n_songs = checkpoint['n_songs']
        self.user_id_map = checkpoint['user_id_map']
        self.song_id_map = checkpoint['song_id_map']

        self.model = NeuralCollaborativeFiltering(
            self.n_users, self.n_songs
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {filepath}")
