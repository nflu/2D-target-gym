import sys
sys.path.insert(0, '/home/trustlap41/gym-target/openai_modified/')
import argparse
import results_plotter_terminal
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file_name", type=str)
	parser.add_argument("num_timesteps", type=int)
	parser.add_argument("task", type=str)
	args = parser.parse_args()
	with open(args.file_name) as f:
		for line in f:
			results_plotter_terminal.plot_results([line], args.num_timesteps, results_plotter_terminal.X_TIMESTEPS, args.task) 

if __name__ == '__main__':
	main()