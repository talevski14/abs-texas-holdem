import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('championship_10000.csv')
df2 = pd.read_csv('championship_10000_2.csv')
df3 = pd.read_csv('championship_10000_3.csv')

df = pd.concat([df1, df2, df3], ignore_index=True)

players = pd.concat([
    df[['agent1', 'agent1 sum of rewards']].rename(columns={'agent1': 'player', 'agent1 sum of rewards': 'rews'}),
    df[['agent2', 'agent2 sum of rewards']].rename(columns={'agent2': 'player', 'agent2 sum of rewards': 'rews'})
])

players['rews_per_game'] = players['rews'] / 10000
players["agent_id"] = players["player"].str.split(":").str[0]

player_stats = players.groupby('agent_id').agg(
    total_rewards=('rews', 'sum'),
    average_rewards=('rews_per_game', 'mean')
).reset_index()

# best_player = player_stats.loc[player_stats['total_rewards'].idxmax(), 'agent_id']
#
# sns.barplot(data=player_stats, x='agent_id', y='total_rewards', palette='viridis')
# plt.title('Total Rewards per Player')
# plt.ylabel('Total Rewards')
# plt.xlabel('Player')
# plt.show()

sns.lineplot(
    data=player_stats,
    x='agent_id',
    y='average_rewards',
    marker='o',
    linewidth=2.5,
    color='green'
)
plt.ylabel('Просечна награда')
plt.xlabel('Агент')
plt.show()

print(player_stats)