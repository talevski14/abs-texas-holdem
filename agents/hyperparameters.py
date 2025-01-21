import torch

SEED = 1626
EPS_TEST = 0.05
EPS_TRAIN = 0.5
BUFFER_SIZE = 100_000
LR = 1e-3
GAMMA = 0.9
N_STEP = 7
TARGET_UPDATE_FREQ = 3_000
EPOCH = 500
STEP_PER_EPOCH = 2000
STEP_PER_COLLECT = 20
UPDATE_PER_STEP = 0.4
BATCH_SIZE = 256
HIDDEN_SIZES = [256, 256, 256, 256, 256]
TRAINING_NUM = 16
TEST_NUM = 8
LOGDIR = "log"
RENDER = 1
WIN_RATE = 1
MIN_WINS = 600
REWARDS_AVERAGE = 50
WATCH = False
AGENT_ID = 2
RESUME_PATH = "log/self-play/ddqn/policy1/policy1.pth"
OPPONENT_PATH = "log/texas-holdem-unlimited/ddqn/policy1/policy1.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"