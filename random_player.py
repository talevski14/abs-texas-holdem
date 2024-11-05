def choose_move(env, agent, mask):
    return env.action_space(agent).sample(mask)