import numpy as np
import random

class SumTree:
    """
    SumTree structure for Prioritized Experience Replay.
    Stores priorities in a binary tree for efficient O(log N) sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree with capacity leaves requires 2*capacity - 1 nodes.
        # We start indices at 0, so parent is at (i-1)//2, children at 2*i+1, 2*i+2
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Priority exponent (0 = no priority, 1 = full priority)
        self.capacity = capacity
        self.max_priority = 1.0  # Initial priority for new transitions

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences.
        beta: Importance-sampling exponent (0 = no IS, 1 = full IS). 
              Should anneal to 1.0 during training.
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normalize for stability

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        """
        Update priorities based on TD errors.
        """
        for idx, error in zip(idxs, errors):
            # Clipping priority to avoid zero or huge values
            p = np.power(np.abs(error) + 1e-5, self.alpha)
            self.tree.update(idx, p)
            
            # Keep track of max priority for new additions
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries
