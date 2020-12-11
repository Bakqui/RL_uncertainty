import random
import torch
from collections import deque

def get_batch(experiences):
    assert len(experiences[0]) == 5
    batch_state = torch.cat([ex[0] for ex in experiences], 0).float()
    batch_act = torch.as_tensor([ex[1] for ex in experiences]).unsqueeze(1)
    batch_reward = torch.as_tensor([ex[2] for ex in experiences]).unsqueeze(1)
    batch_next = torch.cat([ex[3] for ex in experiences], 0).float()
    batch_mask = 1 - torch.as_tensor([ex[4] for ex in experiences]).int().unsqueeze(1)
    return batch_state, batch_act, batch_reward, batch_next, batch_mask

class ReplayBuffer:
    def __init__(self, size: int):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def push(self, ex):
        self.memory.append(ex)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)
