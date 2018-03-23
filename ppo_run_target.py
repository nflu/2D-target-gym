from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy
from baselines import bench
from openai_modified import pposgd_simple_render
import gym
import gym_target
import argparse
from  openai_modified import results_plotter_terminal


def train(env_id, num_timesteps, render_timesteps, render_time, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 64, num_hid_layers = 2)
	logger.configure()
	env = bench.Monitor(gym.make("target-v0"), logger.get_dir())
	pposgd_simple_render.learn(env, policy_fn, 
		max_timesteps = num_timesteps, render_timesteps = render_timesteps,
		render_time = render_time, gamma=1.0, timesteps_per_actorbatch = 5000,
		clip_param=0.10, entcoeff=0.02,
            optim_epochs=10, optim_stepsize=5e-4, optim_batchsize=64,
             lam=0.95, schedule='linear'
        )
	dir  = logger.get_dir()
	with open("scripts/data/data_directories.txt", 'a') as file:
		file.write(dir+"\n")
	results_plotter_terminal.plot_results([dir], num_timesteps, results_plotter_terminal.X_TIMESTEPS, "Target Set Gridworld") 
	#for this to have at least 99 episodes. See line 20 in rolling_window in results_plotter.py from open ai baselines 
	#If there are too few samples, that dimension will be negative and numpy will cause this to crash
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