from constants import *
import math
from scipy.stats import norm

class Team:
    def __init__(self, name, mean_bias=0, rd_bias=0, sigma_bias=0):
        self.name = name
        self.mean = init_rating + mean_bias
        self.rd = init_deviation + rd_bias
        self.sigma = init_variability + sigma_bias

    def __str__(self):
        return f'Team("{self.name}")'

    def get_rating(self):
        return self.mean

    def get_deviation(self):
        return self.rd

    def get_variability(self):
        return self.sigma

    def update_parameters(self, opponent, result):
        # Cap the difference
        result = self.mute_result(result)

        # Calculate game significance
        proximity_adjustment = opponent.mean/(opponent.mean + p5_bias)

        # Calculate parameters of result distribution
        mean_difference = self.mean - opponent.mean
        outcome_std = math.sqrt(self.rd**2 + self.sigma**2 + opponent.rd**2 + opponent.sigma**2)
        win_prob = 1 - norm.cdf(0, loc=mean_difference, scale=outcome_std)

        # Update mean
        result_adjustment = (result - mean_difference) / outcome_std
        learning_weight = mean_adjustment_rate * self.rd**2 / outcome_std**2
        self.mean = self.mean + proximity_adjustment * learning_weight * result_adjustment

        # Update sigma
        observed_variance = (result-mean_difference)**2
        self.sigma = math.sqrt(self.sigma**2 + sigma_adjustment_rate * (observed_variance - self.sigma**2))

        # Update rating deviation
        self.rd = math.sqrt(self.rd**2/(1+deviation_decay_factor))

    def predict_outcome(self, opponent):
        mean_difference = self.mean - opponent.mean
        outcome_std = math.sqrt(self.rd ** 2 + self.sigma ** 2 + opponent.rd ** 2 + opponent.sigma ** 2)

        # Compute win probability for Team A
        return 1 - norm.cdf(0, loc=mean_difference, scale=outcome_std)

    def print_stats(self):
        print(f"{self.name}:")
        print(f"\t- Mean Skill = {self.mean}")
        print(f"\t- Skill Variance = {self.sigma**2}")
        print(f"\t- Rating Uncertainty = {self.rd**2}")

    @staticmethod
    def mute_result(result):
        abs_res = math.fabs(result)
        sign_result = abs_res/result

        adj_result = min(abs_res, max_diff)
        return adj_result * sign_result
