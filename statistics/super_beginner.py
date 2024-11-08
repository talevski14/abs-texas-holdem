def evaluate_hand(observation):
    for index in range(len(observation)):
        if observation[index] == 1:
            if index + 13 < len(observation) and observation[index + 13] == 1:
                return 100
    return 0

def choose_move(observation, mask):
    if evaluate_hand(observation) == 100:
        if mask[2] == 1:
            return 2
        elif mask[1] == 1:
            return 1
    else:
        for index in range(len(mask)):
            if mask[index] == 1:
                return index
            else:
                return None
