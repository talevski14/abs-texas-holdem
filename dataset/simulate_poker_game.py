import pandas as pd
from pettingzoo.classic import texas_holdem_no_limit_v6

class PokerReflexiveAgent:
    def __init__(self):
        pass

    def evaluate_hand(self, observation):
        all_cards = observation["observation"][:52]

        values = [index % 13 for index, value in enumerate(all_cards) if value == 1]
        suits = [index // 13 for index, value in enumerate(all_cards) if value == 1]
        unique_values = set(values)

        value_counts = {value: values.count(value) for value in unique_values}
        counts = list(value_counts.values())

        sorted_values = sorted(unique_values)
        for i in range(len(sorted_values) - 4):
            if sorted_values[i + 4] - sorted_values[i] >= 4:
                if len(set(suits[i:i + 5])) == 1:
                    if sorted_values[i + 4] == 12:
                        return 1.0  # Royal Flush
                    return 0.95  # Straight Flush

        if 4 in counts:
            return 0.90  # Four of a Kind

        if 3 in counts and 2 in counts:
            return 0.85  # Full House

        if len(values) >= 3 and len(set(suits)) == 1:
            return 0.80  # Flush

        for i in range(len(sorted_values) - 4):
            if sorted_values[i + 4] - sorted_values[i] == 4:
                return 0.75  # Straight

        if 3 in counts:
            return 0.7  # Three of a Kind

        if counts.count(2) == 2:
            return 0.65  # Two Pairs

        if 2 in counts:
            return 0.6  # Pair

        if max(values) >= 10:
            return 0.5  # High Card
        else:
            return 0.3  # Low Card Hand

    def decide_action(self, observation):
        hand_strength = self.evaluate_hand(observation)
        available_moves = [index for index, value in enumerate(observation["action_mask"]) if value == 1]

        thresholds = {
            0.9: [4, 3, 2, 1, 0],
            0.8: [3, 2, 1, 0],
            0.7: [2, 1, 0],
            0.5: [1, 0],
            0.0: [0]
        }

        for threshold, moves in thresholds.items():
            if hand_strength >= threshold:
                for move in moves:
                    if move in available_moves:
                        return move
        return None


def simulate_poker_game(rounds=10000):
    env = texas_holdem_no_limit_v6.env(num_players=2)
    env.reset()

    agent1 = PokerReflexiveAgent()
    agent2 = PokerReflexiveAgent()

    dataset = []

    for _ in range(rounds):
        env.reset()

        last_states = {}
        last_actions = {}
        last_rewards = {}
        last_terminations = {}
        states = {}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                if agent == "player_0":
                    action = agent1.decide_action(observation)
                else:
                    action = agent2.decide_action(observation)

            env.step(action)

            state = [int(obs) for obs in observation["observation"]]

            if not action:
                action = "None"
                states[agent] = state

            if agent in last_states:
                dataset.append({
                    "State": last_states[agent],
                    "Action": last_actions[agent],
                    "Reward": last_rewards[agent],
                    "Next_state": state,
                    "Done": last_terminations[agent],
                })

            last_states[agent] = state
            last_actions[agent] = action
            last_rewards[agent] = reward
            last_terminations[agent] = termination or truncation

        for agent in last_states:
            dataset.append({
                "State": last_states[agent],
                "Action": last_actions[agent],
                "Reward": last_rewards[agent],
                "Next_state": states[agent],
                "Done": last_terminations[agent],
            })

    df = pd.DataFrame(dataset)
    df.to_csv("poker_dataset.csv", index=False)

    print(f"Dataset saved with {len(dataset)} entries as poker_dataset.csv.")


if __name__ == "__main__":
    simulate_poker_game(10_000)
