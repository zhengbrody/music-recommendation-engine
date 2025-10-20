"""Plotly Dash dashboard for music recommendation visualization."""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
print("Loading data for dashboard...")
songs_df = pd.read_csv(RAW_DATA_DIR / 'songs.csv')
users_df = pd.read_csv(RAW_DATA_DIR / 'users.csv')
interactions_df = pd.read_csv(PROCESSED_DATA_DIR / 'interactions_processed.csv')

# Prepare statistics
genre_counts = songs_df['genre'].value_counts()
user_activity = interactions_df.groupby('user_id').size().reset_index(name='interaction_count')
song_popularity = interactions_df.groupby('song_id').agg({
    'rating': ['count', 'mean']
}).reset_index()
song_popularity.columns = ['song_id', 'play_count', 'avg_rating']
song_popularity = song_popularity.merge(songs_df[['song_id', 'title', 'artist', 'genre']], on='song_id')

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🎵 Music Recommendation Dashboard", className="text-center mb-4 mt-4"),
            html.Hr()
        ])
    ]),

    # Statistics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(users_df):,}", className="text-primary"),
                    html.P("Total Users")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(songs_df):,}", className="text-success"),
                    html.P("Total Songs")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(interactions_df):,}", className="text-info"),
                    html.P("Total Interactions")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{interactions_df['rating'].mean():.2f}", className="text-warning"),
                    html.P("Avg Rating")
                ])
            ])
        ], width=3),
    ], className="mb-4"),

    # Genre Distribution
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Genre Distribution")),
                dbc.CardBody([
                    dcc.Graph(
                        id='genre-chart',
                        figure=px.pie(
                            values=genre_counts.values,
                            names=genre_counts.index,
                            title="Songs by Genre",
                            hole=0.4
                        )
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Top Genres by Interactions")),
                dbc.CardBody([
                    dcc.Graph(
                        id='genre-interactions',
                        figure=px.bar(
                            interactions_df.merge(songs_df[['song_id', 'genre']], on='song_id')
                            .groupby('genre').size().reset_index(name='count')
                            .sort_values('count', ascending=False),
                            x='genre',
                            y='count',
                            title="Genre Popularity"
                        )
                    )
                ])
            ])
        ], width=6),
    ], className="mb-4"),

    # User Activity Distribution
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("User Activity Distribution")),
                dbc.CardBody([
                    dcc.Graph(
                        id='user-activity',
                        figure=px.histogram(
                            user_activity,
                            x='interaction_count',
                            nbins=50,
                            title="Distribution of User Interactions",
                            labels={'interaction_count': 'Number of Interactions'}
                        )
                    )
                ])
            ])
        ], width=12),
    ], className="mb-4"),

    # Top Songs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Top 20 Most Popular Songs")),
                dbc.CardBody([
                    dcc.Graph(
                        id='top-songs',
                        figure=px.bar(
                            song_popularity.nlargest(20, 'play_count'),
                            x='play_count',
                            y='title',
                            orientation='h',
                            title="Most Played Songs",
                            hover_data=['artist', 'genre', 'avg_rating'],
                            labels={'play_count': 'Play Count', 'title': 'Song Title'}
                        ).update_layout(height=600)
                    )
                ])
            ])
        ], width=12),
    ], className="mb-4"),

    # Song Features Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Song Audio Features by Genre")),
                dbc.CardBody([
                    html.Label("Select Feature:"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[
                            {'label': 'Energy', 'value': 'energy'},
                            {'label': 'Danceability', 'value': 'danceability'},
                            {'label': 'Valence (Positivity)', 'value': 'valence'},
                            {'label': 'Acousticness', 'value': 'acousticness'},
                            {'label': 'Tempo', 'value': 'tempo'}
                        ],
                        value='energy',
                        className="mb-3"
                    ),
                    dcc.Graph(id='feature-by-genre')
                ])
            ])
        ], width=12),
    ], className="mb-4"),

    # Rating Distribution
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Rating Distribution")),
                dbc.CardBody([
                    dcc.Graph(
                        id='rating-dist',
                        figure=px.histogram(
                            interactions_df,
                            x='rating',
                            nbins=5,
                            title="Distribution of User Ratings",
                            labels={'rating': 'Rating'}
                        )
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("User Demographics")),
                dbc.CardBody([
                    dcc.Graph(
                        id='user-demographics',
                        figure=px.histogram(
                            users_df,
                            x='age',
                            nbins=30,
                            title="User Age Distribution",
                            labels={'age': 'Age'}
                        )
                    )
                ])
            ])
        ], width=6),
    ], className="mb-4"),

    # Interactive Recommendation Explorer
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("User Recommendation Explorer")),
                dbc.CardBody([
                    html.Label("Enter User ID:"),
                    dcc.Input(
                        id='user-id-input',
                        type='number',
                        placeholder='Enter user ID',
                        value=0,
                        className="mb-3"
                    ),
                    html.Button('Get User Info', id='user-info-btn', n_clicks=0,
                               className="btn btn-primary mb-3"),
                    html.Div(id='user-info-output')
                ])
            ])
        ], width=12),
    ], className="mb-4"),

], fluid=True)


# Callbacks
@app.callback(
    Output('feature-by-genre', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_feature_chart(selected_feature):
    """Update feature by genre chart."""
    fig = px.box(
        songs_df,
        x='genre',
        y=selected_feature,
        title=f'{selected_feature.capitalize()} by Genre',
        labels={selected_feature: selected_feature.capitalize()}
    )
    return fig


@app.callback(
    Output('user-info-output', 'children'),
    Input('user-info-btn', 'n_clicks'),
    State('user-id-input', 'value')
)
def show_user_info(n_clicks, user_id):
    """Show user information and listening history."""
    if n_clicks == 0 or user_id is None:
        return html.Div("Enter a user ID and click the button to see user information.")

    user = users_df[users_df['user_id'] == user_id]
    if len(user) == 0:
        return html.Div(f"User {user_id} not found.", className="text-danger")

    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    user_songs = user_interactions.merge(songs_df, on='song_id')

    user_info = user.iloc[0]

    # Calculate statistics
    top_genres = user_songs['genre'].value_counts().head(5)
    avg_rating = user_songs['rating'].mean()

    return html.Div([
        html.H5(f"User {user_id} Profile", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.P([html.Strong("Age: "), f"{user_info['age']}"]),
                html.P([html.Strong("Country: "), f"{user_info['country']}"]),
                html.P([html.Strong("Premium: "), f"{'Yes' if user_info['premium'] else 'No'}"]),
            ], width=6),
            dbc.Col([
                html.P([html.Strong("Total Interactions: "), f"{len(user_interactions)}"]),
                html.P([html.Strong("Average Rating: "), f"{avg_rating:.2f}"]),
                html.P([html.Strong("Favorite Genre: "), f"{top_genres.index[0] if len(top_genres) > 0 else 'N/A'}"]),
            ], width=6),
        ]),
        html.Hr(),
        html.H6("Top Genres"),
        dcc.Graph(
            figure=px.bar(
                x=top_genres.values,
                y=top_genres.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Genre'},
                title=f"User {user_id}'s Genre Preferences"
            )
        ),
        html.H6("Recent Listening History (Top 10)", className="mt-3"),
        dbc.Table.from_dataframe(
            user_songs.nlargest(10, 'rating')[['title', 'artist', 'genre', 'rating']],
            striped=True,
            bordered=True,
            hover=True,
            size='sm'
        )
    ])


if __name__ == '__main__':
    print(f"\nDashboard running on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)
