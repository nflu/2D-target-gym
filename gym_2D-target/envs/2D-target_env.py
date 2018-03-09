import numpy as np
import gym 
from gym import spaces 

class 2D-target_env(gym.Env):

	def __init__(self):
		
		# defines size of environment, dimension, and time step size
		self.stepSize = 0.5 
		self.xlim = np.array([-200.0, 200.0])
		self.ylim = np.zeros([-200.0, 200.0])
		self.stateDim = 3 #simple version only takes steps for now
		#must be at least 3, (x, y, lambda)

		self.targetPoint = self.randomPoint()
		def l(point):
			return np.linalg.norm(point - targetPoint)

		self.target = l
		self.state = self.randomPoint()
		self.action_space = spaces.Discrete(5) #left, right, down, up, don't move
		self.curr_step = 0
		self.lastLambda = 0 

	def randomPoint(self):
		point = np.zeros(2)
		self.point[0] = np.random.uniform(xlim[0], xlim[1])
		self.point[1] = np.random.uniform(ylim[0], ylim[1])
		return point

	def randomState(self):
		point = self.randomPoint()
		if (stateDim == 3):
			return np.append(point, self.target(point))
		else:
			return np.append(point, np.zeros(stateDim - 3), self.target(point))

	def _step(self, action):
		self._take_action(action)
		self.curr_step += 1
		reward = self.state[-1] - self.lastLambda
		ob = np.append(self.state, self.targetPoint)
		episode_over = reward == 0 
		return ob , reward, episode_over, {}


	def _take_action(self, action):
		if action != 5:
			self.state[action/2] += 2((action%2)-0.5)*stepSize
		self.lastLambda = self.state[-1]
		self.state[-1] = np.min(self.state[-1], target(self.state[:2]))
