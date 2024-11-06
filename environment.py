from pettingzoo.classic import texas_holdem_no_limit_v6

import random_player, super_beginner

env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)

num_episodes = 100_000

games_won_player1 = 0
games_won_player2 = 0

for episode in range(num_episodes):
    reward_player1 = 0
    reward_player2 = 0
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
            observation = observation["observation"]

            if index % 2 == 0:
                action = random_player.choose_move(env, agent, mask)
            else:
                action = super_beginner.choose_move(observation, mask)
        env.step(action)

    if reward_player1 >= reward_player2:
        games_won_player1 += 1
    else:
        games_won_player2 += 1
env.close()

print(f"Player 1 wins: {games_won_player1}")
print(f"Player 2 wins: {games_won_player2}")