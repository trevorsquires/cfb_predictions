import pandas as pd
import numpy as np
import os
from init_rankings import init_rankings


class EloManager:
    def __init__(self, file_path, params, reset=False):
        self.file_path = file_path
        self.reset = reset
        self.params = params
        self.rankings = None
        self.load_rankings()

    def load_rankings(self):
        if not os.path.exists(self.file_path) or self.reset:
            init_rankings(self.file_path)
        self.rankings = pd.read_csv(self.file_path)

    def save_rankings(self):
        self.rankings.to_csv(self.file_path, index=False)

    @staticmethod
    def g(rd):
        return 1 / np.sqrt(1 + (3 * (rd ** 2)) / (np.pi**2))

    @staticmethod
    def expected_score(r1, r2, g_r2):
        return 1 / (1 + 10 ** (-g_r2 * (r1 - r2) / 100))

    @staticmethod
    def margin_multiplier(margin, base=10):
        return np.log(margin + 1) / np.log(base)

    def update(self, results_df, sample_size):
        """Update the ELO and RD values based on match results."""
        pi_squared = np.pi ** 2
        q = self.params['q']

        # Convert the rankings DataFrame to a dictionary indexed by 'team_id'
        updated_ratings = self.rankings.set_index('team_id').to_dict(orient='index')

        # Iterate through each match in results_df
        for _, row in results_df.iterrows():
            # Extract team IDs and scores from the results
            team_1_id = row['team_1_id']
            team_2_id = row['team_2_id']
            team_1_score = row['team_1_score']
            team_2_score = row['team_2_score']

            # If team does not exist in the database, create an entry for it
            if team_1_id not in updated_ratings:
                updated_ratings[team_1_id] = {
                    'team_name': row['team_1_name'],
                    'ELO': 600,
                    'RD': 1000
                }
            if team_2_id not in updated_ratings:
                updated_ratings[team_2_id] = {
                    'team_name': row['team_2_name'],
                    'ELO': 600,
                    'RD': 1000
                }

            # Get the ratings and RDs for both teams
            r1, rd_1 = updated_ratings[team_1_id]['ELO'], updated_ratings[team_1_id]['RD']
            r2, rd_2 = updated_ratings[team_2_id]['ELO'], updated_ratings[team_2_id]['RD']

            # Calculate the g(RD) factors
            g_rd_1 = self.g(rd_1)
            g_rd_2 = self.g(rd_2)

            # Calculate expected scores
            e_1 = self.expected_score(r1, r2, g_rd_2)
            e_2 = self.expected_score(r2, r1, g_rd_1)

            # Determine the actual scores and margin of victory
            s_1 = 1 if team_1_score > team_2_score else 0 if team_1_score < team_2_score else 0.5
            s_2 = 1 - s_1

            # Calculate the margin of victory
            margin = abs(team_1_score - team_2_score)

            # Calculate the margin multiplier using a logarithmic scale
            M = self.margin_multiplier(margin, base=self.params['margin_log_base'])

            # Calculate rating updates with the margin multiplier
            delta_r_1 = M * q * g_rd_2 * (s_1 - e_1) / (
                    1 / rd_1 ** 2 + q ** 2 * g_rd_2 ** 2 * e_1 * (1 - e_1) / pi_squared)
            delta_r_2 = M * q * g_rd_1 * (s_2 - e_2) / (
                    1 / rd_2 ** 2 + q ** 2 * g_rd_1 ** 2 * e_2 * (1 - e_2) / pi_squared)

            # Update the ELO ratings of both teams
            updated_ratings[team_1_id]['ELO'] += delta_r_1
            updated_ratings[team_2_id]['ELO'] += delta_r_2

            # Update RD values: Increment RD after each match
            updated_ratings[team_1_id]['RD'] = np.sqrt(rd_1 ** 2 + self.params['rd_inc'] ** 2)
            updated_ratings[team_2_id]['RD'] = np.sqrt(rd_2 ** 2 + self.params['rd_inc'] ** 2)

            # Decrease the RD after updating ELO ratings (for the next game)
            updated_ratings[team_1_id]['RD'] = max(updated_ratings[team_1_id]['RD'] - self.params['rd_inc'], 0)
            updated_ratings[team_2_id]['RD'] = max(updated_ratings[team_2_id]['RD'] - self.params['rd_inc'], 0)

        # Convert the updated ratings dictionary back to a DataFrame
        self.rankings = pd.DataFrame.from_dict(updated_ratings, orient='index').reset_index().rename(
            columns={'index': 'team_id'})

    def print_rankings(self):
        print(self.rankings)
