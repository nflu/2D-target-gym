import sys
#sys.path.insert(0, '/home/trustlap41/gym-target/')
sys.path.insert(0, '/Users/Neil/gym-target/')
import gym
import gym_target
import numpy as np
import tensorflow as tf
import argparse
from baselines import logger
from baselines.ppo1 import mlp_policy

def policy_fn(name, ob_space, ac_space):
	return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
		hid_size = 32, num_hid_layers = 3)

def play_from_model(task, num_timesteps, restore_file):
	num_timesteps = int(num_timesteps)
	if task == "grid":
		env = gym.make("target-v0")
	elif task == "dynamics":
		env = gym.make("target-dynamics-v0")
	else:
		raise ValueError("task should be either grid or dynamics instead " + task + " was given.")
	ob = env.reset()
	if restore_file:
		with tf.Session().as_default():
			pi=policy_fn("pi", env.observation_space, env.action_space)
			var_list = pi.get_trainable_variables()
			print(var_list)
			saver=tf.train.Saver()
			saver.restore(tf.get_default_session(), restore_file)
			logger.log("Loaded model from {}".format(restore_file))
			var_list = pi.get_trainable_variables()
			print(var_list)
			for _ in range(num_timesteps):
				a, vpred = pi.act(True, ob)
				ob, r, over, d = env._step(a)
				env._render()
				if over:
					env._close()
					ob = env.reset()
			env._close()
	else:
		for _ in range(num_timesteps):
			a = env.action_space.sample()
			ob, r, over, d = env._step(a)
			env._render()
			if over:
				env._close()
				ob = env.reset()
		env._close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_timesteps", type=float)
	parser.add_argument("task", type=str)
	parser.add_argument("-r", "--restore_file", type=str)
	args = parser.parse_args()
	play_from_model(args.task, args.num_timesteps, args.restore_file)

if __name__ == '__main__':
	main()
