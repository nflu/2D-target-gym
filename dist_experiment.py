from ppo_run_save import train
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("num_trials", type=float)
	parser.add_argument("num_timesteps", type=float)
	parser.add_argument("task", type=str, help="grid or dynamics")
	parser.add_argument("save_prefix", type=str, help="prefix of saved model name")
	args = parser.parse_args()
	policies = ["mlp", "sigmoid", "beta"]
	for policy in policies:
		for _ in range(int(args.num_trials)):
			 train(args.num_timesteps, args.save_prefix + "_" + policy + "/", None, args.task, policy, graph=False)

if __name__ == '__main__':
	main()