import numpy as np
import gym 
from gym import spaces

class TargetDynamicsEnv(gym.Env):

	def __init__(self):
		
		# defines size of environment, dimension, and time step size
		self.timeHorizon = 500
		boxSizeX = 100.0
		boxSizeY = 100.0
		self.timestep_size = 0.01
		maxLambda = np.sqrt((2*boxSizeX)**2 + (2*boxSizeY)**2) #maximum distance between any two points in the space
		maxVelocity = maxLambda / self.timestep_size #velocity to move the maximum distance in minimum time
		maxAccel = maxVelocity / self.timestep_size
		self.high = np.array([boxSizeX, boxSizeY, maxVelocity, maxVelocity, maxLambda, boxSizeX, boxSizeY, 2*np.pi, maxAccel])
		#note that this describes the limits on the observation space.
		#right now the observation space is the state vector concatenated with the target set point
		self.stateDim = 5
		#(x, y, x', y', lambda)
		#if the dimension is increased and the new dimensions will be used in the observation
		#the dimension of the box must also increase.
		self.action_space = spaces.Box(low = -self.high[-2:], high = self.high[-2:]) #direction and magnitude
		self.observation_space = spaces.Box(low = -self.high[:self.stateDim+2], high = self.high[:self.stateDim+2], dtype = np.float32)
		self.curr_step = 0

	def randomPoint(self):
		point = np.zeros(2)
		point[0] = np.random.randint(low = -self.high[0], high = self.high[0])
		point[1] = np.random.randint(low = -self.high[1], high = self.high[1])
		return point

	def randomState(self):
		point = self.randomPoint()
		return np.append(np.append(point, np.zeros(self.stateDim - 3)), self.target(point)) #start at stationary point

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
		self.state[0] = self.state[0] + self.timestep_size * self.state[2]
		self.state[1] = self.state[1] + self.timestep_size * self.state[3]
		self.state[2] = self.state[2] + self.timestep_size * np.cos(action[0]) * action[1]
		self.state[3] = self.state[3] + self.timestep_size * np.sin(action[0]) * action[1]
		self.lastLambda = self.state[-1]
		self.state[-1] = min(self.state[-1], self.target(self.state[:2]))

	def _render(self, mode="human", close=False):
		# I have this problem currently https://github.com/openai/gym/issues/893
		# The fix is to use _close() before close() is called  
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(int(2*self.high[0]), int(2*self.high[1]))
			self.viewer.set_bounds(-self.high[0], self.high[0], -self.high[1], self.high[1])
			target = rendering.make_circle(2)
			target.add_attr(rendering.Transform(translation=self.targetPoint))
			target.set_color(0,0,0)
			self.viewer.add_geom(target)
			agent = rendering.make_circle(2)
			self.agent_translation = rendering.Transform(translation=self.state[:2])
			agent.add_attr(self.agent_translation)
			agent.set_color(1,0,0)
			self.viewer.add_geom(agent)
		else:
			self.agent_translation.set_translation(self.state[0], self.state[1])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None

	def _seed(self):
		#TODO
		pass