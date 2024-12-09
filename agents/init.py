from testing import *
from training import *

if __name__ == "__main__":
    result, agent = train_agent()
    watch(agent)