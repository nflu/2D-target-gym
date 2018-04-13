from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy
from baselines import bench
from openai_modified import pposgd_simple_save
import gym
import gym_target
import argparse
from  openai_modified import results_plotter_terminal

#TODO put in some check that save_prefix is valid so ppo doesn't crash
def train(env_id, num_timesteps, save_prefix, restore_file, task, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 32, num_hid_layers = 3)
	logger.configure()
	if task == "grid":
		env = bench.Monitor(gym.make("target-v0"), logger.get_dir())
	elif task == "dynamics":
		env = bench.Monitor(gym.make("target-dynamics-v0"), logger.get_dir())
	else:
		raise ValueError("task should be either grid or dynamics instead " + task + " was given.")
	pposgd_simple_save.learn(env, policy_fn, 
		max_timesteps = num_timesteps, save_model_with_prefix=save_prefix, restore_model_from_file=restore_file, gamma=1.0, timesteps_per_actorbatch = 10000,
		clip_param=0.10, entcoeff=0.02,
            optim_epochs=10, optim_stepsize=1e-4, optim_batchsize=64,
             lam=0.95, schedule='constant', task = task
        )
	dir  = logger.get_dir()
	with open("scripts/data/mac_directories.txt", 'a') as file:
		file.write(dir+"\n")
	with open(dir + "/metadata.txt", 'w') as file:
		file.write(task + "," + str(num_timesteps))
	results_plotter_terminal.plot_results([dir], num_timesteps, results_plotter_terminal.X_TIMESTEPS, "Target Set " + task) 
	#for this to have at least 99 episodes. See line 20 in rolling_window in results_plotter.py from open ai baselines 
	#If there are too few samples, that dimension will be negative and numpy will cause this to crash
	env.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_timesteps", type=float)
	parser.add_argument("task", type=str)
	parser.add_argument("save_prefix", type=str, help="prefix of saved model name")
	parser.add_argument("-r", "--restore_file", type=str)
	args = parser.parse_args()
	train(None, args.num_timesteps, args.save_prefix, args.restore_file, args.task, None)

if __name__ == '__main__':
	main()