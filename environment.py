from pettingzoo.classic import texas_holdem_no_limit_v6

env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)

num_episodes = 100
reward_player1 = 0
reward_player2 = 0

for episode in range(num_episodes):
    env.reset()
    for index, agent in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        if index % 2 == 0:
            reward_player2 += reward
        else:
            reward_player1 += reward

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = 1

        env.step(action)
env.close()

print(f"Player 1 score: {reward_player1}")
print(f"Player 2 score: {reward_player2}")