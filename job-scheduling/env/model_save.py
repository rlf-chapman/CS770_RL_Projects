from ray.rllib.algorithms.algorithm import Algorithm


root_path = '/Users/omer-appleid/ray_results/PPO_2024-05-07_18-20-17/PPO_JobEnvRL_21151_00000_0_2024-05-07_18-20-17/checkpoint_000000'

model = Algorithm.from_checkpoint(root_path)

model.save('job_scheduling_trial_1')