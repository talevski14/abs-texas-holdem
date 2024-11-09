from pettingzoo.classic import tictactoe_v3
from tianshou.env.pettingzoo_env import PettingZooEnv

def get_env(render_mode=None):
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))