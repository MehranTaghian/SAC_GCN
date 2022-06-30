import os
import random
import numpy as np
import torchgraphs as tg
from utils import save_object, load_object
from threading import Thread


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.thread = None

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
        # Waiting for the previous save to complete
        if self.thread is not None:
            self.thread.join()
        self.thread = Thread(target=save_object, args=(self.buffer.copy(), path))
        self.thread.start()

    def load_buffer(self, path):
        # Waiting for the thread to save the object completely
        if self.thread is not None:
            self.thread.join()
        self.buffer = load_object(path)
        self.position = len(self.buffer) % self.capacity
