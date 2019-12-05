import torch
import numpy as np


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
    
    states, actions, rewards, next_states, dones = batch
    
    states_v = torch.from_numpy(states).to(args.device)
    next_states_v = torch.from_numpy(next_states).to(args.device)
    # actions_v = torch.from_numpy(actions).to(args.device)
    rewards_v = torch.from_numpy(rewards.astype(np.float32)).to(args.device)
    # done_mask = torch.ByteTensor(dones).to(args.device)

    state_action_values = net(states_v)[range(len(actions)), actions.squeeze()]
    if args.double:
        next_state_actions = net(next_states_v).max(dim=1)[1]
        next_state_values = tgt_net(next_states_v)[range(len(actions)), next_state_actions]
    else:    
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[dones] = 0.0

    expected_state_action_values = next_state_values.detach() * args.gamma + rewards_v
    
    return loss(state_action_values, expected_state_action_values)


def duel_dqn_conv_grads_div(net):
    for p in net.named_parameters():
        if 'conv' in p[0]:
            p[1].grad.data.div_(np.sqrt(2))


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards, visibs, dist, angle = [], [], [], []
    for game in range(n_games):
        s = env.reset()
        reward = 0
        info = {}
        for t_step in range(t_max):
            action = agent.sample_actions([s], greedy=greedy)[0]
            s, r, done, info = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
        visibs.append(info['visib'])
        dist.append(info['dist'])
        angle.append(info['angle_between_beams'])
        
    return np.mean(rewards), np.mean(visibs), np.mean(dist), np.mean(angle)
