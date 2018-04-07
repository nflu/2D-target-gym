#using the openai plotting script became too much of a pain so made this and it works much better
from scipy import interpolate
import sys
import re
#sys.path.insert(0, '/home/trustlap41/gym-target/openai_modified/')
sys.path.insert(0, '/Users/Neil/gym-target/openai_modified/')
import argparse
import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("directory_file", type=str, help="this is a text file with the directories of data separated by new lines")
	parser.add_argument("-f", "--data_file", type=str, help="Plots data from files with this name in the listed directories. Default is monitor.csv") #optional
	parser.add_argument("-l", "--last_n_plot", type=int, help="Plots data from this many most recent files. (e.g. last_n_plot = 1 plots the most recently saved run)")
    #need to type -f before argument in command line
	args = parser.parse_args()
	if args.data_file == None:
		args.data_file = "monitor.csv"
	with open(args.directory_file) as f:
		l = []
		for line in f:
			line = line.strip() #need to remove whitespace at end of file name
			line += "/" + args.data_file
			l.append(line)
		i = 0
		fig, axes = plt.subplots(args.last_n_plot)
		for file in l[-args.last_n_plot:]:
			exists = True
			try:
				data = get_reward(np.genfromtxt(file, delimiter=',', skip_header=2,
	                     skip_footer=10, names=['r', 'l', 't']))
			except OSError as e:
				print(e)
				exists = False
			if exists:
				task = ""
				try:
					directory = file[:-len(args.data_file)]
					with open(directory + "metadata.txt") as metadata:
						task = re.sub(r',.*', '', metadata.readline())
				except OSError as e:
					pass
				tck = interpolate.splrep(range(len(data)) , data, k=2, s=2e3 * len(data))
				axes[i].plot(data, 'ko', markersize=1)
				axes[i].plot(interpolate.splev(range(len(data)),tck))
				axes[i].set(xlabel='time steps', ylabel='reward')
				title  = task + " " + re.sub(r'.*openai-', '', file)
				axes[i].set_title(title)
				i+=1
		fig.subplots_adjust(hspace=0.5)

	plt.show()

def get_reward(data):
	rewards = np.zeros(len(data))
	for i in range(len(data)):
		rewards[i] = data[i][0]
	return rewards
	
if __name__ == '__main__':
	main()