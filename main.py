from dqn_discrete.dqn_model import DQNModel, DuelDQNModel, DQNAgent, TargetNet
from dqn_discrete.replay_buffer import ReplayBuffer, fill
from dqn_discrete.common_utils import *
from dqn_discrete.discrete_action_wrapper import make_env
from dqn_discrete.args import get_args

import torch.nn as nn
from tqdm import trange
from tensorboardX import SummaryWriter
import os

from interf_dqn.dqn_discrete.common_utils import duel_dqn_conv_grads_div, calc_loss_dqn


class YABuffer(ReplayBuffer):
    def __init__(self, data_dir='exp_data'):
        files = os.listdir(data_dir)
        super().__init__(len(files))

        for file_id in trange(len(files)):
            file = files[file_id]
            data = np.load(os.path.join(data_dir, file))
            self.add(data['state'], data['action'], data['reward'], data['next_state'], data['done'])

    def sample(self, batch_size):
        ob, act, rew, next_ob, dones = super().sample(batch_size)
        ob = shift(ob)
        next_ob = shift(next_ob)
        return ob, act, rew, next_ob, dones


def main():
    writer = SummaryWriter('logs/fully_randomized_model')

    args = get_args()
    print(vars(args))

    env = make_env()
    observation_shape = env.observation_space.shape
    print('obs shape', observation_shape)
    n_actions = env.action_space.n
    hidden_step_size = n_actions + 1 # no op is excluded, plus step number
    hidden_size = 100 * hidden_step_size
    net = DuelDQNModel(observation_shape, n_actions, hidden_step_size).to(args.device)
    #net.load_state_dict(torch.load('/home/dmitry/dev/interf_game/eval_models/dqn_exp_log_loss_diff_actions'))

    #net = DuelDQNModelWalk([100, n_actions + 1], n_actions).to(args.device)
    #net.load_state_dict(torch.load('model'))

    target_net = TargetNet(net)
    exp_replay = ReplayBuffer(args.buffer_size)
    #ya_replay = YABuffer()

    agent = DQNAgent(net, args.init_eps, args.device)
    if args.loss_type == 'MSE':
        loss = nn.MSELoss()
    elif args.loss_type == 'Huber':
        loss = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    state, hidden = env.reset(), np.zeros(shape=hidden_size, dtype=np.float32)
    state, hidden = fill(exp_replay, agent, env, state, hidden, n_steps=args.init_buff_size)

    step_begin = 0
    for step in trange(step_begin, step_begin + int(args.total_steps/args.rollout_steps + 1)):

        agent.epsilon = linear_decay(step, args)

        # play
        state, hidden = fill(exp_replay, agent, env, state, hidden, n_steps=args.rollout_steps)

        # train
        optimizer.zero_grad()
        batch = exp_replay.sample(args.rollout_steps * args.batch_size)
        #ob1, act1, rew1, next_ob1, dones1 = exp_replay.sample(args.rollout_steps * args.batch_size)
        #ob2, act2, rew2, next_ob2, dones2 = ya_replay.sample(50)

        #ob = np.concatenate([ob1, ob2], axis=0)
        #act = np.concatenate([act1, act2])
        #rew = np.concatenate([rew1, rew2])
        #next_ob = np.concatenate([next_ob1, next_ob2], axis=0)
        #done = np.concatenate([dones1, dones2])

        #batch = (ob, act, rew, next_ob, done)

        loss_v = calc_loss_dqn(batch, net=net, tgt_net=target_net.target_model, args=args, loss=loss)
        loss_v.backward()

        if args.duel:
            duel_dqn_conv_grads_div(net)
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), args.grad_norm)
        optimizer.step()

        writer.add_scalar('loss', loss_v.item(), step)
        writer.add_scalar('grad_norm', grad_norm, step)


        ###############
        #optimizer.zero_grad()
        #yabatch = ya_replay.sample(args.rollout_steps * args.batch_size)
        #exp_loss = calc_loss_dqn(yabatch, net=net, tgt_net=target_net.target_model, args=args, loss=loss).detach().cpu()
        #writer.add_scalar('exp_loss', exp_loss, step)
        ###############

        if step % args.target_net_freq == 0:
            # Load agent weights into target_network
            target_net.sync()

        if step % args.eval_freq == 0:
            agent.epsilon = args.eval_eps
            reward, visib, dist, angle = evaluate(make_env(seed=step), agent, hidden_size, n_games=10, greedy=False)

            writer.add_scalar('reword', reward, step)
            writer.add_scalar('visib', visib, step)
            writer.add_scalar('dist', dist, step)
            writer.add_scalar('angle', angle, step)

            agent.epsilon = linear_decay(step, args)
            
            print("Updates: {}, num timesteps: {}, buff size: {}, epsilon: {:.4f}, visib: {:.2f}, reward {:.2f}".format(
                step, int(step * args.rollout_steps), len(exp_replay), agent.epsilon, visib, reward)
            )

            torch.save(net.state_dict(), 'model')


if __name__ == '__main__':
    main()
