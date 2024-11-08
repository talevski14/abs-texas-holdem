from pettingzoo.classic import texas_holdem_no_limit_v6

import super_beginner

env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)

num_episodes = 1_000_000

games_won_player1 = 0
games_won_player2 = 0
rewards = []

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
                action = super_beginner.choose_move(observation, mask)
            else:
                action = super_beginner.choose_move(observation, mask)
        env.step(action)

    if reward_player1 >= reward_player2:
        games_won_player1 += 1
    else:
        games_won_player2 += 1

    rewards.append((reward_player1, reward_player2))
env.close()

print(f"Player 1 wins: {games_won_player1}")
print(f"Player 2 wins: {games_won_player2}")

sum1 = 0
sum2 = 0
for reward in rewards:
    sum1 += reward[0]
    sum2 += reward[1]

average_reward1 = sum1 / num_episodes
average_reward2 = sum2 / num_episodes

print(f"Average reward per episode for player 1: {average_reward1}")
print(f"Average reward per episode for player 2: {average_reward2}")