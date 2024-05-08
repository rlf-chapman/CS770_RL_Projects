import gymnasium as gym
from gymnasium import spaces, vector
import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from functools import partial
from ray.rllib.algorithms.algorithm import Algorithm

from job_env import *

if ray.is_initialized():
  ray.shutdown()
ray.init(num_cpus=4)

config = (PPOConfig()
          .environment(JobEnvRL)
          .framework('torch')
          .training(gamma=0.9)
          .rollouts(num_rollout_workers=3)
)

stop = {"timesteps_total": 12000}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop),
)

tuner.fit()
