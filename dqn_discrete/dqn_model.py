import torch
import torch.nn as nn
import numpy as np
import copy


class DQNModel(nn.Module):
    """A simple DQN net"""
    
    def __init__(self, input_shape, n_actions):
        super(DQNModel, self).__init__()

        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))

    def forward(self, x):
        x = x.float() / 255.
        conv_out = self.conv(x).view(x.shape[0], -1)
        return self.fc(conv_out)


class DuelDQNModel(nn.Module):
    """A Dueling DQN net"""
    
    def __init__(self, input_shape, n_actions, history_step_size):
        super(DuelDQNModel, self).__init__()

        print('history step size = ', history_step_size)

        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape, self.conv)
        self.history_step_size = history_step_size

        #self.history = nn.LSTM(history_step_size, history_step_size)
        #self.lstm_out_size = history_step_size

        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )
        # h_adv = self.fc_adv.register_hook(lambda grad: grad/torch.sqrt(2))
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # h_val = self.fc_val.register_hook(lambda grad: grad/torch.sqrt(2))

    def _get_conv_out(self, shape, conv):
        o = conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))

    def forward(self, x, histo):
        x = x.float() / 255.0

        conv_out = self.conv(x).view(x.shape[0], -1)

        #histo = histo.view([histo.shape[0], -1, self.history_step_size])
        #last_histo_step = histo[:, -1, -2:]
        #last_histo_visib_and_time = last_histo_step[:, -2:]

        #out = torch.cat([conv_out, last_histo_visib_and_time], dim=1)

        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)

        return val + adv - adv.mean()


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


class DQNAgent:
    """
    Simple DQNAgent which calculates Q values from list of observations
                          calculates actions given np.array of qvalues
    """
    def __init__(self, dqn_model, epsilon, device):
        
        self.dqn_model = dqn_model
        self.epsilon = epsilon
        self.device = device

    def get_number_of_actions(self):
        return self.dqn_model.n_actions

    def get_q_values(self, states, hiddens):
        """
        Calculates q-values given list of obseravations
        """
        
        states = self._state_processor(states)
        hiddens = self._state_processor(hiddens)

        q_values = self.dqn_model.forward(states, hiddens)

        return q_values.detach().cpu().numpy()

    def _state_processor(self, states):
        """
        Conversts list of states into torch tensor and copies it to the device
        """
        
        # return torch.tensor(states).to(self.device)
        return torch.from_numpy(np.array(states)).to(self.device)          

    def sample_actions(self, states, hidden, greedy=False):
        """
        Pick actions given array of qvalues
        Uses epsilon-greedy exploration strategy
        """
        
        qvalues = self.get_q_values(states, hidden)
        best_actions = qvalues.argmax(axis=-1)
        if greedy:
            return best_actions
        
        batch_size, n_actions = qvalues.shape
        
        mask = np.random.random(size=batch_size) < self.epsilon
        random_actions = np.random.choice(n_actions, size=sum(mask))        
        best_actions[mask] = random_actions
        
        return best_actions
