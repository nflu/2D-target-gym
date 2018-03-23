import gym
import gym_target
import numpy as np
env = gym.make('target-dynamics-v0')
env._reset()
a = [np.random.uniform()*2*np.pi, 40]
print(a)
i = 0
for _ in range(10000):
	if i >= 1000:
		a = np.zeros(2)
	env._step(a)
	print(env.state)
	env._render()
	i+=1
env._close()