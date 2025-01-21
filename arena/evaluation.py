import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('championship_10000_2.csv')

players = pd.concat([
    df[['agent1', 'agent1 sum of rewards']].rename(columns={'agent1': 'player', 'agent1 sum of rewards': 'rews'}),
    df[['agent2', 'agent2 sum of rewards']].rename(columns={'agent2': 'player', 'agent2 sum of rewards': 'rews'})
])

players['rews_per_game'] = players['rews'] / 10000

player_stats = players.groupby('player').agg(
    total_rewards=('rews', 'sum'),
    average_rewards=('rews_per_game', 'mean')
).reset_index()

best_player = player_stats.loc[player_stats['total_rewards'].idxmax(), 'player']

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
sns.barplot(data=player_stats, x='player', y='total_rewards', palette='viridis')
plt.title('Total Rewards per Player')
plt.ylabel('Total Rewards')
plt.xlabel('Player')

plt.subplot(3, 1, 2)
sns.lineplot(
    data=player_stats,
    x='player',
    y='average_rewards',
    marker='o',
    linewidth=2.5,
    color='green'
)
plt.title('Average Rewards per Game per Player')
plt.ylabel('Average Rewards')
plt.xlabel('Player')

plt.suptitle(f'The Best Player is: {best_player}', fontsize=16, fontweight='bold', color='darkgreen')
plt.tight_layout(rect=(0, 0, 1, 0.95))

plt.show()
