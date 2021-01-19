import sys
sys.path.insert(0,'..')

import numpy
import time
import gym_explore.envs

env = gym_explore.envs.ExploreArcadeGenericEnv(size=8)
state = env.reset()

actions_count  = env.action_space.n

k   = 0.02
fps = 0.0

action = 0
while True:

    if numpy.random.randint(100) < 20:
        action = numpy.random.randint(actions_count)
        
    time_start = time.time()
    state, reward, done, _ = env.step(action)
    time_stop  = time.time()

    env.render()

    fps = (1.0 - k)*fps + k*1.0/(time_stop - time_start + 0.00001)
    print("fps = ", round(fps, 2))