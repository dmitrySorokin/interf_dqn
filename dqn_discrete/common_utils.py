import torch
import numpy as np


def action2vec(action_id, n_actions):
    res = np.zeros(n_actions - 1)
    if action_id < n_actions - 1:
        res[action_id] = 1
    return res


def rescale_imin(imin):
    imin = imin / 255.0 * 2.0 * np.random.uniform(0.98, 1.02) + 1e-3
    return -np.log(imin) / 10


def rescale_visib(visib):
    return -np.log(1.0 - visib + 1e-3) + visib


def shift(state):
    start = np.random.randint(0, len(state))
    result = []
    for i in range(start, start + len(state)):
        result.append(state[i % len(state)])
    return np.asarray(result)


def linear_decay(cur_step, args):
    """
    Returns epsilon value corresponding to the current step
    given nuber of current step and args
    """
    if cur_step >= args.decay_steps:
        return args.final_eps
    return (args.init_eps * (args.decay_steps - cur_step) +
            args.final_eps * cur_step) / args.decay_steps


def calc_loss_dqn(batch, net, tgt_net, args, loss):
    
    states, hiddens, actions, rewards, next_states, next_hiddens, dones = batch
    
    states_v = torch.from_numpy(states).to(args.device)
    hiddens_v = torch.from_numpy(hiddens).to(args.device)

    next_states_v = torch.from_numpy(next_states).to(args.device)
    next_hiddens_v = torch.from_numpy(next_hiddens).to(args.device)

    # actions_v = torch.from_numpy(actions).to(args.device)
    rewards_v = torch.from_numpy(rewards.astype(np.float32)).to(args.device)
    # done_mask = torch.ByteTensor(dones).to(args.device)

    state_action_values = net(states_v, hiddens_v)[range(len(actions)), actions.squeeze()]
    if args.double:
        next_state_actions = net(next_states_v, hiddens_v).max(dim=1)[1]
        next_state_values = tgt_net(next_states_v, next_hiddens_v)[range(len(actions)), next_state_actions]
    else:    
        next_state_values = tgt_net(next_states_v, next_hiddens_v).max(1)[0]
    next_state_values[dones] = 0.0

    expected_state_action_values = next_state_values.detach() * args.gamma + rewards_v
    
    return loss(state_action_values, expected_state_action_values)


def duel_dqn_conv_grads_div(net):
    for p in net.named_parameters():
        if 'conv' in p[0]:
            p[1].grad.data.div_(np.sqrt(2))


def evaluate(env, agent, hidden_size, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards, visibs, dist, angle = [], [], [], []
    h = np.zeros(hidden_size, dtype=np.float32)
    for game in range(n_games):
        s = env.reset()
        reward = 0
        info = {}
        for t_step in range(t_max):
            action = agent.sample_actions([s], [h], greedy=greedy)[0]
            s, r, done, info = env.step(action)
            reward += r

            #imin = rescale_imin(info['imin'])
            rescaled_visib = rescale_visib(info['visib'])
            action_vec = action2vec(action, agent.get_number_of_actions())
            h = np.append(h[2 + len(action_vec):], [*action_vec, rescaled_visib, env.max_steps - env.n_steps]).astype(h.dtype)

            if done:
                h = np.zeros_like(h, dtype=h.dtype)
                break

        rewards.append(reward)
        visibs.append(info['visib'])
        dist.append(info['dist'])
        angle.append(info['angle_between_beams'])
        
    return np.mean(rewards), np.mean(visibs), np.mean(dist), np.mean(angle)
