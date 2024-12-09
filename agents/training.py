import random
from typing import Optional, Tuple
from tianshou.policy import BasePolicy
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
import os
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer

from hyperparameters import *
from environment import get_env
from agents import get_agents

def train_agent(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = DummyVectorEnv([get_env for _ in range(TRAINING_NUM)])
    test_envs = DummyVectorEnv([get_env for _ in range(TEST_NUM)])

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_envs.seed(SEED)
    test_envs.seed(SEED)


    policy, optim, agents = get_agents(
        agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )


    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_step=BATCH_SIZE * TRAINING_NUM)


    log_path = os.path.join(LOGDIR, "texas-holdem-unlimited", "dqn")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)


    def save_best_fn(policy):
        model_save_path = os.path.join(
            LOGDIR, "texas-holdem-unlimited", "dqn", "policy3.pth"
        )
        torch.save(
            policy.policies[agents[AGENT_ID - 1]].state_dict(), model_save_path
        )

    def stop_fn(rewards):
        return rewards >= REWARDS_AVERAGE

    def train_fn(epoch, env_step):
        policy.policies[agents[AGENT_ID - 1]].set_eps(EPS_TRAIN)

    def test_fn(epoch, env_step):
        policy.policies[agents[AGENT_ID - 1]].set_eps(EPS_TEST)

    def reward_metric(rews):
        return rews[:, AGENT_ID - 1]


    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        EPOCH,
        STEP_PER_EPOCH,
        STEP_PER_COLLECT,
        TEST_NUM,
        BATCH_SIZE,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=UPDATE_PER_STEP,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[AGENT_ID - 1]]