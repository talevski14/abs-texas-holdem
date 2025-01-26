import math

import pandas as pd
from agents_list import agents_list


def probability(rating1, rating2):
    return 1.0 / (1 + math.pow(10, (rating1 - rating2) / 400.0))


def elo_rating(Ra, Rb, K, outcome):
    Pb = probability(Ra, Rb)
    Pa = probability(Rb, Ra)

    Ra = Ra + K * (outcome - Pa)
    Rb = Rb + K * ((1 - outcome) - Pb)

    return Ra, Rb

K = 32

ratings = {}
for agent in agents_list:
    ratings[agent["name"]] = 1500

df1 = pd.read_csv("championship_10000.csv")
df2 = pd.read_csv("championship_10000_2.csv")
df3 = pd.read_csv("championship_10000_3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

df['agent1 sum of rewards'] = df['agent1 sum of rewards'].astype(float)
df['agent2 sum of rewards'] = df['agent2 sum of rewards'].astype(float)

for index, row in df.iterrows():
    outcome = ((row["agent1 sum of rewards"] / 10_000) - (-100)) / (100 - (-100))

    agent1_rating, agent2_rating = elo_rating(ratings[row["agent1"]], ratings[row["agent2"]], K, outcome)
    ratings[row["agent1"]] = agent1_rating
    ratings[row["agent2"]] = agent2_rating

for element in sorted(ratings, key=ratings.get, reverse=True):
    print(f'{element} has rating {ratings[element]}')
