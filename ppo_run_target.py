from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy, pposgd_simple
import gym
import gym_target
import argparse


def train(env_id, num_timesteps, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 64, num_hid_layers = 3)
	env = gym.make("target-v0")
	pposgd_simple.learn(env, policy_fn, 
		max_timesteps = num_timesteps,
		gamma=1.0, timesteps_per_actorbatch = 128,
		clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
             lam=0.95, schedule='linear',
        )
	env.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_timesteps", type=float)
	args = parser.parse_args()
	train(None, args.num_timesteps, None)

if __name__ == '__main__':
	main()