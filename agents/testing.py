from typing import Optional
from tianshou.policy import BasePolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector

from environment import get_env
from agents import get_agents
from hyperparameters import *

def watch(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optim, agents = get_agents(
        agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    policy.policies[agents[AGENT_ID - 1]].set_eps(EPS_TEST)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=10, render=RENDER)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, AGENT_ID - 1].mean()}, length: {lens.mean()}")