import pandas as pd
import requests
from constants import endpoints


class Requester:
    def __init__(self, base_url, headers, year, season_type):
        self.base_url = base_url
        self.headers = headers
        self.year = year
        self.season_type = season_type

    def get_statistic(self, endpoint):
        url = f"{self.base_url}{endpoints[endpoint]}?year={self.year}&seasonType=regular"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None