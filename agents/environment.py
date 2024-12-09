from pettingzoo.classic import tictactoe_v3
from pettingzoo.classic import texas_holdem_no_limit_v6
from tianshou.env.pettingzoo_env import PettingZooEnv

def get_env(render_mode=None):
    return PettingZooEnv(texas_holdem_no_limit_v6.env(render_mode=render_mode))