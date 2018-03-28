import sys
#sys.path.insert(0, '/home/trustlap41/gym-target/openai_modified/')
sys.path.insert(0, '/Users/Neil/gym-target/openai_modified/')
import argparse
import results_plotter_terminal
from glob import glob
import os.path as osp
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("file_name", type=str)
	parser.add_argument("num_timesteps", type=int)
	parser.add_argument("task", type=str)
	args = parser.parse_args()
	
	with open(args.file_name) as f:
		l = []
		for line in f:
			line = line.strip() #need to remove whitespace at end of file name
			l.append(line)
		#TODO find the set of tasks and then group files according to task and number of time steps and then plot each like that
		results_plotter_terminal.plot_results(l, args.num_timesteps, results_plotter_terminal.X_TIMESTEPS, args.task) 
	
if __name__ == '__main__':
	main()