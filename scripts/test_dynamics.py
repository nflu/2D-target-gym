import sys
#sys.path.insert(0, '/home/trustlap41/gym-target/')
sys.path.insert(0, '/Users/Neil/gym-target/openai_modified/')
import gym
import gym_target
import numpy as np
env = gym.make('target-dynamics-v0')
env._reset()
i = 0
a = [3*np.pi/2, 400]
for _ in range(10000):
	#if i % 1000 == 0:
		#a = [np.random.uniform()*2*np.pi, 400]
	env._step(a)
	print(env.state)
	env._render()
	i+=1
env._close()