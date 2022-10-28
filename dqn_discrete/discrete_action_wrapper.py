from dqn_discrete.common_utils import shift

import gym
import gym_interf
import numpy as np
from collections import deque
from .common_utils import rescale_visib


RADIUS = 0.714


class BeamEllipticityRandomizer(gym.Wrapper):
    def reset(self):
        rotation = np.random.uniform(0, 2 * np.pi)
        ellip = np.random.uniform(0.7, 1.0)
        self.env.set_beam_rotation(rotation)
        self.env.set_beam_ellipticity(ellip)
        return self.env.reset()


class BeamRadiusRandomizer(gym.Wrapper):
    def reset(self):
        r = RADIUS * np.random.uniform(0.8, 1.2)
        self.env.set_radius(r)
        return self.env.reset()


class ChannelShifter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return shift(self.env.reset())

    def step(self, actions):
        n_back = np.random.randint(0, 8)
        self.env.set_backward_frames(n_back)
        obs, rew, done, info = self.env.step(actions)
        return shift(obs), rew, done, info


class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, e):
        super().__init__(e)
        self.shape = self.env.action_space.shape[0]
        self.ndir = self.shape * 2
        self.step_fractions = [0.01, 0.05, 0.1]
        self.n_actions = self.ndir * len(self.step_fractions) + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def reset(self):
        actions = np.random.uniform(low=-1.0, high=1.0, size=self.shape)
        return self.env.reset(actions=actions)
        #x = np.random.uniform(low=-0.5, high=0.5)
        #y = np.random.uniform(low=-0.5, high=0.5)
        #return self.env.reset([x, y, -2 * x, 2 * y])

    def step(self, action_id: int):
        actions = [0] * self.shape

        if action_id < self.n_actions - 1:

            step_fraction = self.step_fractions[action_id // self.ndir]
            action_id = action_id % self.ndir
            actions[action_id // 2] = step_fraction * (
                -1 if action_id % 2 == 0 else 1
            )

        # change me to add different scales to actions
        actions = np.asarray(actions) * np.ones(self.shape)

        return self.env.step(actions)


class BrightnessRandomizer(gym.Wrapper):
    def __init__(self, e):
        super().__init__(e)

    def randomize(self, obs):
        obs = obs * np.random.uniform(0.7, 1.3)
        obs = np.minimum(obs, 255)
        return obs.astype(np.uint8)

    def reset(self):
        obs = self.env.reset()
        return self.randomize(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.randomize(obs), rew, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        orig_shape = env.observation_space.shape
        self.shape = (k * orig_shape[0], orig_shape[1], orig_shape[2])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        res = np.asarray(self.frames).reshape(self.shape)
        return res


class RewardChanger(gym.Wrapper):
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        #return state, info['visib'] * done - 1.0 / 200.0, done, info
        rescaled_visib = rescale_visib(info['visib'])
        return state, rescaled_visib - 1.0, done, info


class CameraPositionRandomizer(gym.Wrapper):
    def reset(self):
        x_min_rnd = np.random.uniform(0.7, 1.3)
        self.env.set_xmin(-3.57 / 2 * x_min_rnd)
        x_max_rnd = np.random.uniform(0.7, 1.3)
        self.env.set_xmax(3.57 / 2 * x_max_rnd)
        return self.env.reset()


class RealisticBeamsRandomizer(gym.Wrapper):
    def reset(self):
        use_masks = np.random.random() < 0.5
        self.env.use_beam_masks(use_masks)
        return self.env.reset()


def make_env(seed=None):
    env = gym.make('interf-v2')
    env.set_radius(RADIUS)
    env.set_max_steps(100)
    env = DiscreteActionWrapper(env)
    env = BeamRadiusRandomizer(env)
    #env = BeamEllipticityRandomizer(env)
    env = BrightnessRandomizer(env)
    env = ChannelShifter(env)
    #env = FrameStack(env, 3)
    env = RewardChanger(env)
    #env = CameraPositionRandomizer(env)
    #env = RealisticBeamsRandomizer(env)
    env.seed(seed)
    return env
