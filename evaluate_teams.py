from requester import Requester
from local_secrets import headers
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def get_shortest_path(graph, source, target):
    try:
        return nx.shortest_path(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return None


class TeamEvaluator:
    def __init__(self):
        requester = Requester(headers, year=2024, season_type='regular')
        games = requester.get_statistic('games')

        # Initialize an empty dictionary to store the lists for each column
        graph_dict = {
            'loser_id': [],
            'winner_id': [],
            'loser_name': [],
            'winner_name': []
        }

        # Iterate through each game and append values to the respective lists
        for game in games:
            if game['completed'] and game['home_points'] and game['away_points']:
                if game['home_points'] > game['away_points']:  # home team won
                    graph_dict['loser_id'].append(game['away_id'])
                    graph_dict['winner_id'].append(game['home_id'])
                    graph_dict['loser_name'].append(game['away_team'])
                    graph_dict['winner_name'].append(game['home_team'])
                else:  # home team lost
                    graph_dict['loser_id'].append(game['home_id'])
                    graph_dict['winner_id'].append(game['away_id'])
                    graph_dict['loser_name'].append(game['home_team'])
                    graph_dict['winner_name'].append(game['away_team'])

        # Create the DataFrame using the populated dictionary
        graph_df = pd.DataFrame(graph_dict)

        # Create directed graph
        self.G = nx.DiGraph()
        for _, row in graph_df.iterrows():
            self.G.add_edge(row['loser_name'], row['winner_name'])

        self.cycles = list(nx.simple_cycles(self.G))
        self.sccs = list(nx.strongly_connected_components(self.G))

    def plot_graph(self):
        # Create a meta-graph for SCCs
        meta_graph = nx.DiGraph()

        # Create SCC node mapping
        scc_map = {frozenset(scc): f'SCC_{i}' for i, scc in enumerate(self.sccs)}

        # Add SCC nodes to the meta-graph
        for scc in scc_map.keys():
            meta_graph.add_node(scc_map[scc])

        # Add edges between SCCs in the meta-graph
        for u, v in self.G.edges():
            scc_u = scc_map[frozenset(next(scc for scc in scc_map if u in scc))]
            scc_v = scc_map[frozenset(next(scc for scc in scc_map if v in scc))]
            if scc_u != scc_v:  # Add edge only between different SCCs
                meta_graph.add_edge(scc_u, scc_v)

        # Set up plot
        plt.figure(figsize=(10, 6))

        # Draw the meta-graph with node labels
        pos = nx.spring_layout(meta_graph)  # Positions for all nodes in the graph
        nx.draw(meta_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12,
                font_weight='bold', arrowsize=20)

        # Draw the labels for each node
        labels = {node: f'{node}\n{list(next(scc for scc in self.sccs if scc_map[frozenset(scc)] == node))}' for node in
                  meta_graph.nodes()}
        nx.draw_networkx_labels(meta_graph, pos, labels=labels)

        # Show plot
        plt.title("Meta-Graph of SCCs")
        plt.show()

    def compare_teams(self, team1, team2):
        two_team_cycles = [cycle for cycle in self.cycles if team1 in cycle and team2 in cycle]
        path1 = get_shortest_path(self.G, team1, team2)
        path2 = get_shortest_path(self.G, team2, team1)

        if len(two_team_cycles) > 0:
            min_cycle = min(two_team_cycles, key=len)
            print(f"The two teams are equivalent because of the cycle {min_cycle}")
            return min_cycle
        elif path1:
            print(f"{team2} is better than {team1} because of the path {path1}")
            return path1
        elif path2:
            print(f"{team1} is better than {team2} because of the path {path2}")
            return path2
        else:
            print('The two teams are incomparable')
            return None

    def find_equiv_teams(self, team):
        team_cycles = [cycle for cycle in self.cycles if team in cycle]
        equiv_teams = {team for cycle in team_cycles for team in cycle}
        print(f"The equivalent teams to {team} are {equiv_teams}")
        return equiv_teams

    def topological_rankings(self):
        meta_graph = nx.DiGraph()
        scc_to_id = {frozenset(scc): f'SCC_{i}' for i, scc in enumerate(self.sccs)}
        id_to_scc = {f'SCC_{i}': scc for i, scc in enumerate(self.sccs)}

        for scc in scc_to_id.keys():
            meta_graph.add_node(scc_to_id[scc])
        for u, v in self.G.edges():
            scc_u = scc_to_id[frozenset(next(scc for scc in scc_to_id if u in scc))]
            scc_v = scc_to_id[frozenset(next(scc for scc in scc_to_id if v in scc))]
            if scc_u != scc_v:  # Add edge only between different SCCs
                meta_graph.add_edge(scc_u, scc_v)
        scc_order = list(nx.topological_sort(meta_graph))

        # Print SCC ranks along with the names of the teams
        print("Rankings:")
        count = 1
        for scc in reversed(scc_order):
            if count > 100:
                continue
            teams = id_to_scc[scc]
            if len(teams) > 1:
                print(f"T{count}: {[print(f'{name}, ', end='') for name in teams]}")
            else:
                print(f"{count}: {list(teams)[0]}")
            count += len(teams)


evaluator = TeamEvaluator()
# evaluator.compare_teams('Clemson', 'Michigan')

evaluator.topological_rankings()
# evaluator.plot_graph()
