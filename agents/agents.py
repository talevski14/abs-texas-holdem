from typing import Optional, Tuple
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
import gymnasium
from tianshou.utils.net.common import Net
from copy import deepcopy

from hyperparameters import *
from environment import get_env

def get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    state_shape = observation_space.shape or observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    if agent_learn is None:
        # model
        net = Net(
            state_shape,
            action_shape,
            hidden_sizes=HIDDEN_SIZES,
            device=DEVICE,
        ).to(DEVICE)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=LR)
        agent_learn = DQNPolicy(
            net,
            optim,
            GAMMA,
            N_STEP,
            target_update_freq=TARGET_UPDATE_FREQ,
            is_double=False,
        )
        if RESUME_PATH:
            agent_learn.load_state_dict(torch.load(RESUME_PATH))

    if agent_opponent is None:
        if OPPONENT_PATH:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(OPPONENT_PATH))
        else:
            agent_opponent = RandomPolicy()

    if AGENT_ID == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents