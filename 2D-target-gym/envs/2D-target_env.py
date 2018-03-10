import numpy as np
import gym 
from gym import spaces

class target_2D_env(gym.Env):

	def __init__(self):
		
		# defines size of environment, dimension, and time step size
		self.timeHorizon = 500
		self.stepSize = 1
		boxSizeX = 100
		boxSizeY = 100
		self.high = np.array([boxSizeX, boxSizeY])
		self.stateDim = 3 #simple version only takes steps for now
		#must be at least 3 : (x, y, lambda)
		#if the dimension is increased and the new dimensions will be used in the observation
		#the dimension of the box must also increase.

		self.action_space = spaces.Discrete(5) #left, right, down, up, don't move
		self.observation_space = spaces.Box(low = -self.high, high = self.high, dtype = np.float32)

	def randomPoint(self):
		point = np.zeros(2)
		self.point[0] = np.random.randint(low = -self.high[0], high = self.high[0]) #for now this will be gridworld
		self.point[1] = np.random.randint(low = -self.high[0], high = self.high[0])
		return point

	def randomState(self):
		point = self.randomPoint()
		if (stateDim == 3):
			return np.append(point, self.target(point))
		else:
			return np.append(point, np.zeros(stateDim - 3), self.target(point))

	def _get_reward(self):
		return self.state[-1] - self.lastLambda

	def _step(self, action):
		self._take_action(action)
		self.curr_step += 1
		reward = self._get_reward()
		ob = self.state[:len(high)] 
		episode_over = self.curr_step >= self.timeHorizon
		return ob , reward, episode_over, {}

	def _reset(self):
		self.targetPoint = self.randomPoint()
		def l(point):
			return np.linalg.norm(point - targetPoint)

		self.target = l
		self.state = self.randomState()
		self.curr_step = 0
		self.lastLambda =  float("inf")
		ob = np.append(self.state, self.targetPoint)
		return ob

	def _take_action(self, action):
		if action != 5:
			self.state[action/2] += 2((action%2)-0.5)*stepSize
		self.lastLambda = self.state[-1]
		self.state[-1] = np.min(self.state[-1], target(self.state[:2]))