from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import gym
import gym_target
import argparse
from  openai_modified import results_plotter_terminal


def train(env_id, num_timesteps, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 64, num_hid_layers = 2)
	logger.configure()
	env = bench.Monitor(gym.make("target-v0"), logger.get_dir())
	trpo_mpi.learn(env, policy_fn, 
		max_timesteps = num_timesteps, gamma=1.0, timesteps_per_batch = 5000,
		max_kl=0.01, entcoeff=0.02,
            vf_iters=5, vf_stepsize=1e-3,
             lam=0.95, cg_iters=10, cg_damping=0.1,
        )
	dir  = logger.get_dir()
	with open("data/data_directories.txt", 'a') as file:
		file.write(dir+"\n")
	results_plotter_terminal.plot_results([dir], num_timesteps, results_plotter_terminal.X_TIMESTEPS, "Target Set Gridworld") 
	#for this to have at least 99 episodes. See line 20 in rolling_window in results_plotter.py from open ai baselines 
	#If there are too few samples, that dimension will be negative and numpy will cause this to crash
	env.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_timesteps", type=float)
	args = parser.parse_args()
	train(None, args.num_timesteps, None)

if __name__ == '__main__':
	main()