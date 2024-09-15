from requester import Requester
from local_secrets import *
import pandas as pd

# Initialization
base_url = 'https://api.collegefootballdata.com/'
requester = Requester(base_url, headers, year=2018, season_type='regular')

'''Roster Analysis'''
roster = requester.get_statistic('team_roster')
rosters_df = pd.DataFrame(roster)
team = 'Michigan'
team_roster = rosters_df[rosters_df.team == team].reset_index()

player_stats = requester.get_statistic('player_season_stats')
player_stats_df = pd.DataFrame(player_stats)
player_stats_pivot_df = player_stats_df.pivot(index=['playerId'], columns=['category', 'statType'], values='stat').reset_index()
player_stats_pivot_df.columns = ['_'.join(col).strip() for col in player_stats_pivot_df.columns.values]
player_stats_pivot_df.rename(columns={'playerId_': 'id'}, inplace=True)

team_stats = team_roster.merge(player_stats_pivot_df, how='inner', on='id')
rbs = team_stats[team_stats['position'] == 'RB'][['first_name', 'last_name', 'rushing_CAR', 'rushing_LONG', 'rushing_TD', 'rushing_YDS']]
wrs = team_stats[team_stats['position'] == 'WR'][['first_name', 'last_name', 'receiving_LONG', 'receiving_REC', 'receiving_TD', 'receiving_YDS', 'receiving_YPR']]

