# This code is taken from OpenAi baselines
# from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
from tqdm import trange
from .common_utils import action2vec, rescale_imin, rescale_visib


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, hidden, action, reward, obs_tp1, hidden_p1, done):
        data = (obs_t, hidden, action, reward, obs_tp1, hidden_p1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
    def _encode_sample(self, idxes):
        obses_t, hiddens, actions, rewards, obses_tp1, hiddens_p1, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, hidden, action, reward, obs_tp1, hidden_p1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            hiddens.append(np.array(hidden, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            hiddens_p1.append(np.array(hidden_p1, copy=False))
            dones.append(done)

        return np.array(obses_t), \
               np.array(hiddens), \
               np.array(actions), \
               np.array(rewards), \
               np.array(obses_tp1), \
               np.array(hiddens_p1),\
               np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        
        if len(self._storage) <= batch_size:
            return self._storage
        # Warning: replace=False makes random.choice O(n)
        idxes = np.random.choice(len(self._storage), batch_size, replace=True)
        return self._encode_sample(idxes)


def fill(buffer, agent, env, s_begin, h_begin, n_steps=1):
    s, h = s_begin, h_begin

    iter = range
    if n_steps > 1000:
        iter = trange

    for _ in iter(n_steps):
        assert h.dtype == np.float32

        action = agent.sample_actions([s], [h])[0]
        next_s, r, done, info = env.step(action)

        #imin = rescale_imin(info['imin'])
        visib = rescale_visib(info['visib'])
        action_vec = action2vec(action, agent.get_number_of_actions())
        next_h = np.append(h[2 + len(action_vec):], [visib, env.max_steps - env.n_steps, *action_vec]).astype(h.dtype)

        buffer.add(s, h, action, r, next_s, next_h, done)
        if done:
            s = env.reset()
            h = np.zeros_like(h, dtype=h.dtype)
        else:
            s = next_s
            h = next_h

    return s, h
