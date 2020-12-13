import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


model_dict = {'dqn': 'DQN',
              'boot': 'BootDQN',
              'bbb': 'Bayes by Backprop',
              'drop': 'MC Dropout',
              'en': 'Deep Ensemble'}

# data_dict = ['DQN', 'BootDQN', 'Bayes by Backprop', 'MC Dropout', 'Deep Ensemble']
# data_dict = ['DQN', 'BootDQN', 'MC Dropout', 'Deep Ensemble']
data_dict = ['DQN', 'BootDQN', 'MC Dropout']

def plot(args):
    period = args.period
    load_dir = args.load_dir
    if not os.path.exists(load_dir):
        os.mkdir(load_dir)
    if args.save_dir is None:
        save_dir = args.load_dir
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.figure(figsize=(10, 6))
    for i, agent in enumerate(data_dict):
        data = np.load(os.path.join(load_dir, '%s.npy' % agent))
        rst = np.zeros((data.shape[0], data.shape[1]-period+1))
        for j in range(data.shape[0]):
            rst[j] = moving_average(data[j], period)
        avg = rst.mean(0)[:200]
        std = rst.std(0)[:200]
        idx = np.arange(len(avg))
        color = 'C%d' % i
        print(color)
        plt.plot(idx, avg, color=color, linewidth=3, label=agent)
        plt.fill_between(idx, avg+0.5*std, avg-0.5*std,
                         color=color, alpha=0.1)

    plt.xlabel('Episodes', fontsize=15)
    plt.xlabel('Avg. rewards', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.title('CartPole-v1 Scores: 10 reps', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'rst.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--period', default=10)
    parser.add_argument('--load_dir', default='rst/k=4;max_norm=5')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    plot(args)
