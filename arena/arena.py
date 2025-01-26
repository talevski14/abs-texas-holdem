import pandas as pd
from pettingzoo.classic import texas_holdem_no_limit_v6
from typing import Optional, Tuple, Any
from tianshou.policy import BasePolicy, DQNPolicy, RandomPolicy, MultiAgentPolicyManager
import gymnasium
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.data import Collector
import torch

from agents_list import agents_list


def get_env(render_mode=None):
    return PettingZooEnv(texas_holdem_no_limit_v6.env(render_mode=render_mode))


def get_agents(
        agent1: Optional[BasePolicy] = None,
        agent2: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()

    agents = [agent1, agent2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def play(
        agent1: Optional[BasePolicy] = None,
        agent2: Optional[BasePolicy] = None,
        episodes: int = 100,
        eps_test1: Optional[float] = None,
        eps_test2: Optional[float] = None,
) -> tuple[Any, Any]:
    env = DummyVectorEnv([lambda: get_env(render_mode="ansi")])
    policy, optim, agents = get_agents(
        agent1=agent1, agent2=agent2
    )
    policy.eval()
    policy.policies[agents[0]].set_eps(eps_test1)
    policy.policies[agents[1]].set_eps(eps_test2)

    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=episodes, render=0)
    rews, lens = result["rews"], result["lens"]
    return rews, lens


def load_agent(agent):
    state_shape, action_shape = load_environment()

    agent1 = torch.load(agent['path'])
    model1 = Net(state_shape, action_shape, hidden_sizes=agent['hidden_sizes'])
    model1.load_state_dict(agent1, strict=False)
    policy_agent_1 = DQNPolicy(model1, torch.optim.Adam(model1.parameters(), lr=agent['lr']), agent['gamma'],
                               agent['n_step'])
    return policy_agent_1


def load_environment():
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    state_shape = observation_space.shape or observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    return state_shape, action_shape


if __name__ == "__main__":
    results = []
    counter = 0
    for agent1 in agents_list:
        for agent2 in agents_list:
            if agent1 is agent2:
                continue
            player = load_agent(agent1)
            opponent = load_agent(agent2)

            rews, lens = play(player, opponent, 10000, agent1['eps_test'],
                              agent2['eps_test'])
            counter += 1
            print(counter)

            results.append({
                'agent1': agent1['name'],
                'agent2': agent2['name'],
                'agent1 sum of rewards': rews[:, 0].sum(),
                'agent2 sum of rewards': rews[:, 1].sum(),
                'game length average': lens.mean(),
            })

    df = pd.DataFrame(results)
    df.to_csv("championship_10000_3.csv", index=False)
