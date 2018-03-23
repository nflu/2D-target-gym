import results_plotter_terminal
import argparse 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file_name", type=str)
	parser.add_argument("num_timesteps", type=int)
	parser.add_argument("task", type=str)
	args = parser.parse_args()
	with open(args.file_name) as f:
		results_plotter_terminal.plot_results(f, args.num_timesteps, results_plotter_terminal.X_TIMESTEPS, args.task) 

if __name__ == '__main__':
	main()