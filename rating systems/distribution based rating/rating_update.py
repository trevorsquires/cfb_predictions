from src.requester import Requester
import pandas as pd
from local_secrets import *
from constants import *
from team import Team


def record_game(game, verbose=False):
    team1 = team_ratings.get(game['home_team'])
    team2 = team_ratings.get(game['away_team'])
    if not game['completed']:
        return  # don't update if game hasn't been played yet
    if team1 is None or team2 is None:
        return  # if either team is unknown, don't do an update

    # Print pregame ratings
    if verbose:
        print("Pregame ratings")
        team1.print_stats()
        team2.print_stats()

        # Predict result
        print(f"{team1} has a {team1.predict_outcome(team2) * 100}% chance to beat {team2}")

    # Observe result
    result = game['home_points'] - game['away_points']

    # Print results
    if verbose:
        print(f"{team1} scored {game['home_points']} points and {team2} scored {game['away_points']}")

    # Update ratings for both teams
    team1.update_parameters(team2, result)
    team2.update_parameters(team1, -1 * result)

    # Print postgame ratings
    if verbose:
        print("Postgame ratings")
        team1.print_stats()
        team2.print_stats()


def calculate_initial_mean_bias(team, ap_dict):
    poll_bias = ap_dict.get(team['school'], -ap_poll_weight*ap_baseline)/ap_poll_weight + ap_baseline
    conference_bias = p5_bias if team['conference'] in p5_conferences else p5_bias/2
    total_bias = max(poll_bias, conference_bias)
    return total_bias

requester = Requester(headers, year=2024, season_type='regular')
teams = requester.get_statistic('fbs_teams')
all_polls = requester.get_statistic('rankings')
ap_rankings = all_polls[0]['polls'][2]['ranks']
ap_dict = {team['school']: team['points'] for team in ap_rankings}


team_ratings = {}
# Create list of teams
for team in teams:
    mean_bias = calculate_initial_mean_bias(team, ap_dict)
    team_ratings[team['school']] = Team(team['school'], mean_bias)


# Update team ratings by iterating through games played
games = requester.get_statistic('games')
num_weeks = 14
for week in list(range(1, num_weeks+1)):
    # Filter down into week
    week_games = [game for game in games if game['week'] == week]
    for game in week_games:
        print_results = (game['home_team'] == team_of_interest) or (game['away_team'] == team_of_interest)
        record_game(game, verbose=print_results)

# Print the rankings
data = []
for team_name, team_obj in team_ratings.items():
    team_data = {
        "Team": team_name,
        "Rating": team_obj.get_rating(),
        "Variance": team_obj.get_variability(),
        "Deviation": team_obj.get_deviation(),
    }
    data.append(team_data)

# Create the DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by="Rating", ascending=False)
df = df.reset_index(drop=True)
df["Rank"] = df.index + 1
print(df)


# Predict conference championships
for game in games_of_interest:
    print(f"Game: {game[0]} vs {game[1]}")
    team1 = team_ratings[game[0]]
    team2 = team_ratings[game[1]]

    print("Pregame ratings")
    team1.print_stats()
    team2.print_stats()

    # Predict result
    print(f"{team1} has a {team1.predict_outcome(team2) * 100}% chance to beat {team2}\n")
