from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy
import pposgd_simple_render
import gym
import gym_target
import argparse


def train(env_id, num_timesteps, render_timesteps, render_time, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 64, num_hid_layers = 3)
	env = gym.make("target-v0")
	pposgd_simple_render.learn(env, policy_fn, 
		max_timesteps = num_timesteps, render_timesteps = render_timesteps,
		render_time = render_time, gamma=1.0, timesteps_per_actorbatch = 300,
		clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
             lam=0.95, schedule='linear',
        )
	env.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_timesteps", type=float)
	parser.add_argument("render_timesteps", type=float , help="will render these last many timesteps")
	parser.add_argument("render_time", type=float , help="will render each frame for at least this much time")
	args = parser.parse_args()
	train(None, args.num_timesteps, args.render_timesteps, args.render_time, None)

if __name__ == '__main__':
	main()