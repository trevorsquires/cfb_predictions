from requester import Requester
import pandas as pd
from local_secrets import *


def init_rankings(ratings_file_name):


    base_url = 'https://api.collegefootballdata.com/'
    requester = Requester(base_url, headers, year=2024, season_type='regular')
    teams = requester.get_statistic('fbs_teams')

    # Creating first elo
    default_elo = 800
    all_polls = requester.get_statistic('rankings')
    ap_rankings = all_polls[0]['polls'][2]['ranks']
    ap_dict = {item['school']: default_elo + 500 - 8*(item['rank']-1) for item in ap_rankings}
    power_conf = {
        'SEC': 1050,
        'ACC': 1000,
        'Big Ten': 1050,
        'Big 12': 1000,
        'Pac-12': 1000
    }

    ratings_df = pd.DataFrame(
        {
            'team_id': [team['id'] for team in teams],
            'team_name': [team['school'] for team in teams],
            'ELO': [ap_dict.get(team['school'], power_conf.get(team['conference'], default_elo)) for team in teams],
            'RD': [1000] * len(teams)
        }
    )

    ratings_df.to_csv(ratings_file_name, index=False)

