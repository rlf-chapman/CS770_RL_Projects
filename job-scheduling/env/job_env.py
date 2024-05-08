import numpy as np
import gymnasium as gym
from gymnasium import spaces
from job_scheduling import JobEnv

class JobEnvRL(JobEnv, gym.Env):
    def __init__(self, seed=None):
        n_workers = 10
        n_parts = 10
        JobEnv.__init__(self, seed=seed, n_days=100, n_workers=n_workers)
        gym.Env.__init__(self)
        self.max_jobs = 20
        self.max_offers = 7
        n_obs_jobs = self.max_jobs + self.max_offers
        self.reset(seed)
        self.action_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self.max_offers + self.max_jobs)])
        n_obs_jobs = self.max_jobs + self.max_offers
        self.observation_space = spaces.Dict({'period': spaces.Box(low=0, high=self.n_days, dtype=np.float32),
                                              'job_reqs': spaces.Box(low=0, high=self.n_workers, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_parts': spaces.Box(low=0, high=n_parts, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_soft_deadline': spaces.Box(low=-2*n_parts, high=2*n_parts, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_hard_deadline': spaces.Box(low=0, high=2*n_parts, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_comps': spaces.Box(low=0, high=1, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_payments': spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_length': spaces.Box(low=0, high=n_parts, shape=(n_obs_jobs,), dtype=np.float32),
                                              'job_rate': spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs_jobs,), dtype=np.float32)
                                             })

    def reset(self, seed=None, options=None):
        self.prev_pay = 0
        super().reset()
        return self.get_observation(), {}

    def soft_reset(self):
        super().soft_reset()

    def step(self, actions):
        work_actions = list(actions[:self.max_jobs])
        job_acceptances = self.truncate_acceptances(list(actions[self.max_jobs:]))
        super().step(job_acceptances, work_actions, record=False)
        reward = self.total_payment - self.prev_pay
        complexity_bonus = np.mean(self.create_job_array('complication_probability', np.float32)) * 0.1  # Adjust weight
        worker_utilization = np.mean(self.create_job_array('n_workers', np.float32)) * 0.2   # Access through env
        reward += complexity_bonus + worker_utilization
        self.prev_pay = self.total_payment
        done = self.current_day >= self.n_days
        truncated = False
        return self.get_observation(), reward, done, truncated, {}

    def truncate_acceptances(self, accepts):
        n_jobs = len(self.jobs)
        for i in range(len(self.offers)):
            if n_jobs >= self.max_jobs:
                accepts[i] = 0
            elif accepts[i] == 1:
                n_jobs += 1
        return accepts

    def create_attribute_array(self, job_list, attr, max_length, dtype, is_method=False):
        if is_method:
            methods = [getattr(job, attr) for job in job_list]
            attr_values = [method() for method in methods]
        else:
            attr_values = [getattr(job, attr) for job in job_list]
        padded_values = attr_values + [0] * (max_length - len(attr_values))
        return np.array(padded_values, dtype=dtype)

    def create_job_array(self, attr, dtype, is_method=False):
        return np.concatenate([
            self.create_attribute_array(self.jobs, attr, self.max_jobs, dtype, is_method),
            self.create_attribute_array(self.offers, attr, self.max_offers, dtype, is_method)
        ])

    def get_observation(self):
        obs = {'period': np.array([self.current_day], dtype=np.float32),
               'job_reqs': self.create_job_array('n_workers', np.float32),
               'job_parts': self.create_job_array('parts', np.float32),
               'job_comps': self.create_job_array('complication_probability', np.float32),
               'job_soft_deadline': self.create_job_array('soft_deadline', np.float32),
               'job_hard_deadline': self.create_job_array('hard_deadline', np.float32),
               'job_payments': self.create_job_array('payment', np.float32),
               'job_length': self.create_job_array('expected_length', np.float32, is_method=True),
               'job_rate': self.create_job_array('return_rate', np.float32, is_method=True)
              }
        return obs

    def render(self, mode='human'):
        print(self.get_pretty_str())