from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy, pposgd_simple
import gym
import  #TODO this might need to be updated

def train(env_id, num_timesteps, seed):
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
			hid_size = 64, num_hid_layers = 2)
	env = gym.make()

def main():
#TODO