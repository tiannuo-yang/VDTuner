import sys 
sys.path.append("..") 

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv


if __name__ == '__main__':
    # prepare the environment
    env = RealEnv()
    model = PollingBayesianOptimization(env, seed=1)
    
    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for i in range(200-7):
        model.step()
