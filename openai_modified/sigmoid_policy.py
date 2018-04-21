from baselines.ppo1 import mlp_policy
from scipy.special import expit

class SigmoidPolicy(mlp_policy.MlpPolicy):

    def __init__(self, name, *args, **kwargs):
        super(SigmoidPolicy, self).__init__(name, *args, **kwargs)

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        self.ac_space = ac_space
        super(SigmoidPolicy, self)._init(ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var)

    def act(self, stochastic, ob):
        ac , vpred = super(SigmoidPolicy, self).act(stochastic, ob)
        ac = ac * (self.ac_space.high-self.ac_space.low) + self.ac_space.low
        return ac, vpred