import numpy as np
import gym 
from gym import spaces

class TargetEnv(gym.Env):

	def __init__(self):
		
		# defines size of environment, dimension, and time step size
		self.timeHorizon = 500
		self.stepSize = 1
		boxSizeX = 100.0
		boxSizeY = 100.0
		maxLambda = np.sqrt((2*boxSizeX)**2 + (2*boxSizeY)**2)
		self.high = np.array([boxSizeX, boxSizeY, maxLambda, boxSizeX, boxSizeY])
		#note that this describes the limits on the observation space.
		#right now the observation space is the state vector concatenated with the target set point
		self.stateDim = 3 #simple version only takes steps for now
		#must be at least 3 : (x, y, lambda)
		#if the dimension is increased and the new dimensions will be used in the observation
		#the dimension of the box must also increase.
		self.action_space = spaces.Discrete(5) #left, right, down, up, don't move
		self.observation_space = spaces.Box(low = -self.high, high = self.high, dtype = np.float32)

	def randomPoint(self):
		point = np.zeros(2)
		point[0] = np.random.randint(low = -self.high[0], high = self.high[0]) #for now this will be gridworld
		point[1] = np.random.randint(low = -self.high[0], high = self.high[0])
		return point

	def randomState(self):
		point = self.randomPoint()
		if (self.stateDim == 3):
			return np.append(point, self.target(point))
		else:
			return np.append(point, np.zeros(stateDim - 3), self.target(point))

	def _get_reward(self):
		return self.lastLambda - self.state[-1] #formulation is negated so that reward can be maximized

	def _step(self, action):
		self._take_action(action)
		self.curr_step += 1
		reward = self._get_reward()
		ob = np.append(self.state, self.targetPoint)
		episode_over = self.curr_step >= self.timeHorizon
		return ob , reward, episode_over, {}

	def _reset(self):
		self.targetPoint = self.randomPoint()
		def l(point):
			return np.linalg.norm(point - self.targetPoint)

		self.target = l
		self.state = self.randomState()
		self.curr_step = 0
		self.lastLambda =  float("inf")
		self.viewer = None
		ob = np.append(self.state, self.targetPoint)
		return ob

	def _take_action(self, action):
		if action != 5:
			self.state[action//2] += 2 * ((action % 2)-0.5)*self.stepSize 
		self.lastLambda = self.state[-1]
		self.state[-1] = min(self.state[-1], self.target(self.state[:2]))

	def _render(self, mode="human", close=False):
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(2*self.high[0], 2*self.high[1])
			self.viewer.set_bounds(-self.high[0], self.high[0], -self.high[1], self.high[1])
			target = rendering.make_circle(1)
			target.add_attr(rendering.Transform(translation=self.targetPoint))

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _seed(self):
		#TODO
		pass
