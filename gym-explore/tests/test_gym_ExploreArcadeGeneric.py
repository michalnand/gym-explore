import gym
import gym_explore
import numpy
import time

#create 8x8 rooms
env     = gym.make("ExploreArcadeGeneric-v0", size=4)
state   = env.reset()

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

    env.render(True)
    time.sleep(0.01)

    fps = (1.0 - k)*fps + k*1.0/(time_stop - time_start + 0.00001)
    print("fps = ", round(fps, 2))