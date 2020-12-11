import gym
import numpy as np
import os
import random
import torch
import torch.nn as nn

from argparse import ArgumentParser
from models.dqn import DQNAgent, BootDQNAgent
from models.bayes import B3DQNAgent
from models.mcdropout import DropDQNAgent
from models.deepen import EnDQNAgent

agent_dict = {'DQN': DQNAgent,
              'BootDQN': BootDQNAgent,
              'Bayes by Backprop': B3DQNAgent,
              'MC Dropout': DropDQNAgent,
              'Deep Ensemble': EnDQNAgent}

def greedy(agent, state):
    action = agent(state)
    if isinstance(action, tuple):
        action = action[0]
    return np.argmax(action.detach().numpy())

def majority_vote(agent, state):
    qvalues = agent(state)
    actions = torch.argmax(qvalues, dim=1)
    return actions.mode().values.item()

def epsilon_greedy(agent, epsilon, state):
    if random.random() < epsilon:
        return random.randint(0, agent.a_dim - 1)
    else:
        action = agent(state)
        return np.argmax(action.detach().numpy())

def thompson_sampling(agent, epsilon, state):
    q_mu, q_sigma = agent(state)
    q_estimate = q_mu + epsilon*torch.randn_like(q_sigma)*q_sigma
    return np.argmax(q_estimate.detach().numpy())

exploration_dict = {'DQN': epsilon_greedy,
                    'BootDQN': epsilon_greedy,
                    'Bayes by Backprop': thompson_sampling,
                    'MC Dropout': thompson_sampling,
                    'Deep Ensemble': thompson_sampling}

exploitation_dict = {'DQN': greedy,
                     'BootDQN': majority_vote,
                     'Bayes by Backprop': greedy,
                     'MC Dropout': greedy,
                     'Deep Ensemble': greedy}

activation_dict = {'relu': nn.ReLU,
                   'leaky_relu': nn.LeakyReLU,
                   'elu': nn.ELU,
                   'tanh': nn.Tanh}

def get_action(agent_name, agent, state, epsilon=None):
    if epsilon is not None:
        return exploration_dict[agent_name](agent, epsilon, state)
    else:
        return exploitation_dict[agent_name](agent, state)

def main(args):
    model_dir = args.model_dir
    save_dir = args.save_dir
    max_episode = args.max_episode
    max_epi_step = args.max_epi_step
    agent_name = args.agent_name
    model_save_dir = os.path.join(model_dir, agent_name)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    epoch_reward = []

    for epoch in range(args.n_epoch):

        epsilon = args.epsilon
        epsilon_min = args.epsilon_min
        decay_rate = args.decay_rate

        lamb = args.lamb
        lamb_max = args.lamb_max
        increase_rate = args.increase_rate

        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        h_act = activation_dict[args.h_act]

        agent = agent_dict[agent_name](s_dim=state_dim,
                                       a_dim=action_dim,
                                       h_dim=args.h_dim,
                                       h_act=h_act,
                                       buffer_size=args.buffer_size,
                                       batch_size=args.batch_size,
                                       lr=args.lr,
                                       gamma=args.gamma,
                                       theta=args.theta,
                                       dropout=args.dropout,
                                       weight_decay=args.weight_decay,
                                       noise_level=args.noise_level,
                                       n_sample=args.n_sample,
                                       n_model=args.n_model)

        print('\nEpoch %d\n' % (epoch+1))
        reward_list = []

        for episode in range(max_episode):

            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)

            reward_epi = []
            loss_epi = []
            action = None
            agent.training = True
            for epi_step in range(max_epi_step):
                if agent_name == 'BootDQN':
                    agent.current_head = np.random.randint(agent.n_model)

                # make an action based on epsilon greedy
                action = get_action(agent_name, agent, state, epsilon)

                before_state = state

                state, reward, done, _ = env.step(action)

                state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)

                # make a transition and save to replay memory
                transition = [before_state, action, reward, state, done]
                agent.save_memory(transition)

                if agent.train_start():
                    agent.training = True
                    if agent_name == 'Bayes by Backprop':
                        loss = agent.train(lamb, k=args.k, max_norm=args.max_norm)
                    else:
                        loss = agent.train(k=args.k, max_norm=args.max_norm)
                    loss_epi.append(loss)
                if done:
                    break

            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)
            agent.training = False
            for epi_step in range(max_epi_step):

                # make an greedy action
                action = get_action(agent_name, agent, state)

                before_state = state

                state, reward, done, _ = env.step(action)

                state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)
                reward_epi.append(reward)

                if done:
                    break

            if epsilon > epsilon_min:
                epsilon -= decay_rate
            else:
                epsilon = epsilon_min

            if lamb > lamb_max:
                lamb += increase_rate
            else:
                lamb = lamb_max

            reward_list.append(sum(reward_epi))

            if (episode+1) % 10 == 0:
                print('Episode:%d \t Rewards:%.1f \t Epsilon:%.2f' % (episode+1, reward_list[-1], epsilon))

        torch.save(agent.q_net.state_dict(),
                   os.path.join(model_save_dir, '%d.pt' % epoch))
        epoch_reward.append(reward_list)

    env.close()
    arr = np.asarray(epoch_reward)
    np.save(os.path.join(save_dir, agent_name), arr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent_name', type=str, default='DQN')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--model_dir', type=str, default='experiments')
    parser.add_argument('--save_dir', type=str, default='rst')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--max_episode', type=int, default=210)
    parser.add_argument('--max_epi_step', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--h_act', type=str, default='elu')
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--theta', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--n_model', type=int, default=5)
    parser.add_argument('--n_sample', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--epsilon_min', type=float, default=5e-3)
    parser.add_argument('--decay_rate', type=float, default=5e-3)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--lamb', type=float, default=1e-5)
    parser.add_argument('--lamb_max', type=float, default=1e-3)
    parser.add_argument('--increase_rate', type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
