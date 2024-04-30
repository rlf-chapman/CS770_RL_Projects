import gymnasium as gym
from csbsingleenv import *
import time
import numpy as np


env = CodersStrikeBackSingle()
fps =  env.metadata.get('video.frames_per_second')

# Set pseudorandom seed for the getting the same game
env.seed(1234)

for n in range(10):
    state, _ = env.reset()

    for i in range(1,10000):
        # display the game
        render = env.render()
        if not render:
            break
        # Take action (simple policy)
        targetX, targetY = state[6:8]
        thrust = 100
        action = np.array([targetX, targetY, thrust], dtype=np.float32)
        # Do a game step
        state, reward, done, trunc, _ = env.step(action)

        # Print the state

        print('---------- Tick %d' % i)
        print('Pod angle ', state[0])
        print('Pod (x,y): (%d, %d)' % (state[1], state[2]))
        print('Pod velocity (v_x,v_y): (%d, %d)' % (state[3], state[4]))
        print('First Checkpoint (x,y): (%d, %d)' % (state[5],state[6]))
        print('Second Checkpoint (x,y): (%d, %d)' % (state[7],state[8]))
        print('Reward ', reward)
        print('Total Time', env.time)
        print('done ', done)

        # Slow down the cycle for a realistic simulation
        time.sleep(1.0/fps)

        # The game end if the flag done is True
        if done:
            break
env.close()

