import gym
import gym_target
env = gym.make('target-v0')
env._reset()
for _ in range(1):
	a = env.action_space.sample()
	print(a)
	env._step(a)
	print(env.state)
	for _ in range(100):
		env._render()
print("before close")
env._close()