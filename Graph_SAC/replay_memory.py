import os
import pickle, gzip, pickletools
import random
import numpy as np
import torchgraphs as tg

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*batch)
        state = tg.GraphBatch.collate(state)
        action = np.stack(action)
        reward = np.stack(reward)
        next_state = tg.GraphBatch.collate(next_state)
        done = np.stack(done)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, path):
        path = os.path.join(path, 'buffer.pkl')
        with gzip.open(path, "wb") as f:
            pickled = pickle.dumps(self.buffer)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)

    def load_buffer(self, path):
        with open(path, 'rb') as f:
            p = pickle.Unpickler(f)
            self.buffer = p.load()
            self.position = len(self.buffer) % self.capacity

        # TODO: uncomment the following lines after the buffer is saved as gzip
        # with gzip.open(path, 'rb') as f:
        #     self.buffer = pickle.Unpickler(f)
        #     self.position = len(self.buffer) % self.capacity
