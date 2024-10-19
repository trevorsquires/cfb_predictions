from requester import Requester
from local_secrets import *
import pandas as pd
from elo_handler import EloManager

# TODO need a factor for updating elo MORE when it's a big game, i.e. a non FCS opponent

# Initialize tools
base_url = 'https://api.collegefootballdata.com/'
requester = Requester(headers, year=2024, season_type='regular')
params = {
    'margin_log_base': 5,  # controls how much emphasis you put on MOV
    'rd_inc': 100,  # controls how quickly your RD value decreases
    'q': 3/100  # controls how much results matter
}

elo = EloManager('first_ratings.csv', params, reset=True)
num_weeks = 4

# Grab results
games = requester.get_statistic('games')

for week in list(range(1, num_weeks+1)):
    # Filter down into week
    week_games = [game for game in games if game['week'] == week]

    # Prep df
    results_df = pd.DataFrame(
        {
            'team_1_name': [game['home_team'] for game in week_games],
            'team_2_name': [game['away_team'] for game in week_games],
            'team_1_id': [game['home_id'] for game in week_games],
            'team_2_id': [game['away_id'] for game in week_games],
            'team_1_score': [game['home_points'] for game in week_games],
            'team_2_score': [game['away_points'] for game in week_games],
        }
    )

    # Do the update
    elo.update(results_df, sample_size=week)

print(elo.rankings.sort_values(by='ELO', ascending=False).head(25))




