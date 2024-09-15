from requester import Requester
from local_secrets import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Initialization
base_url = 'https://api.collegefootballdata.com/'
requester = Requester(base_url, headers, year=2023, season_type='regular')

# Collect Data
game_results = requester.get_statistic('games')
team_season_stats = requester.get_statistic('team_season_stats')

# Prep Data for model
games_df = pd.DataFrame(game_results)
team_stats_df = pd.DataFrame(team_season_stats)

# Pivot the team stats dataframe to have each stat in its own column
team_stats_pivot_df = team_stats_df.pivot(index=['season', 'team'], columns=['statName'], values=['statValue']).reset_index()

# Clean the column names (remove prefixes, etc.)
team_stats_pivot_df.columns = ['_'.join(col).strip() for col in team_stats_pivot_df.columns.values]
team_stats_pivot_df.columns = team_stats_pivot_df.columns.str.replace('statValue', '', regex=False)
team_stats_pivot_df.columns = team_stats_pivot_df.columns.str.replace('_', '', regex=False)

# Merge team stats for home_team (team1) and away_team (team2) with suffixes
input_df = pd.merge(games_df, team_stats_pivot_df, how='inner', left_on=['home_team'], right_on=['team'], suffixes=('', '_team1'))
input_df = pd.merge(input_df, team_stats_pivot_df, how='inner', left_on=['away_team'], right_on=['team'], suffixes=('', '_team2'))

# Drop redundant 'team' columns
input_df = input_df.drop(columns=['team_team2'])
input_df = input_df.drop(
    columns=[
        'id', 'season', 'week', 'start_date', 'start_time_tbd', 'completed', 'attendance', 'venue_id', 'venue',
        'excitement_index', 'highlights', 'notes', 'season_team1', 'season_type',
        'home_id', 'home_line_scores', 'home_post_win_prob', 'home_postgame_elo',
        'away_id', 'away_line_scores', 'away_post_win_prob', 'away_postgame_elo'
    ]
)


# Prepare Data for the Model

# Define X (features) and y (target variables)
X = input_df.drop(columns=['home_points', 'away_points'])  # Features are all columns except target
y_home = input_df['home_points']  # Target for home_score
y_away = input_df['away_points']  # Target for away_score

categorical_columns = X.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Convert to string before label encoding
    label_encoders[col] = le  # Store encoders for later use if needed

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Optional: Feature Scaling (if needed for XGBoost)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost model for home_score
home_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
home_xgb_model.fit(X_train, y_home_train)

# XGBoost model for away_score
away_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
away_xgb_model.fit(X_train, y_away_train)

# Predictions with XGBoost
y_home_pred_xgb = home_xgb_model.predict(X_test)
y_away_pred_xgb = away_xgb_model.predict(X_test)

# Evaluate XGBoost models using Mean Squared Error
home_mse_xgb = mean_squared_error(y_home_test, y_home_pred_xgb)
away_mse_xgb = mean_squared_error(y_away_test, y_away_pred_xgb)

print(f"XGBoost - Home Score MSE: {home_mse_xgb}")
print(f"XGBoost - Away Score MSE: {away_mse_xgb}")
