import torch
import torch.nn as nn
from pytorch_util import BinaryTreeLSTMCell

class TreeNode:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.start == self.end

class SampleIncidenceRow(nn.Module):
    def __init__(self, input_size, n, size_g, leaf_prob, left_prob, right_prob, temperature=1.0, max_nonzero=11, min_nonzero=2):
        super().__init__()
        self.n = n
        self.input_size = input_size
        self.size_g = size_g
        self.leaf_prob = leaf_prob
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.temperature = temperature  # Gumbel-Softmax temperature
        self.max_nonzero = max_nonzero
        self.min_nonzero = min_nonzero

        # MLPs for decision making
        self.mlp_left = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.mlp_right = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.mlp_leaf = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # LSTM for updating h
        self.lstm = nn.LSTMCell(input_size, input_size)

        # BinaryTreeLSTMCell for combining states from left and right children
        self.tree_lstm_cell = BinaryTreeLSTMCell(input_size)

    def relaxed_bernoulli(self, logits):
        uniform_noise = torch.rand_like(logits)
        disturbance = 0.1
        gumbel_noise = -torch.log(-torch.log(uniform_noise + disturbance) + disturbance)
        return torch.sigmoid((logits + gumbel_noise) / self.temperature)

    def build_tree(self, start, end):
        node = TreeNode(start, end)
        if start == end:
            return node
        mid = (start + end) // 2
        node.left = self.build_tree(start, mid)
        node.right = self.build_tree(mid + 1, end)
        return node

    def traverse_tree(self, node, h, c, sample=None):
        if sample is None:
            sample = torch.zeros(self.n, device=h.device)

        if node.is_leaf():
            nonzero_count = sample.nonzero().size(0)
            if nonzero_count == self.max_nonzero:
                return sample, h, c
            leaf_logits = self.mlp_leaf(h)
            binary_leaf = torch.bernoulli(leaf_logits)
            sample[node.start] = binary_leaf
            return sample, h, c

        # Initialize left and right states as None
        h_left, c_left, h_right, c_right = None, None, None, None

        left_logits = self.mlp_left(h)
        left_prob = self.relaxed_bernoulli(left_logits)

        if left_prob > self.left_prob:
            h_left, c_left = self.lstm(h.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
            h_left, c_left = h_left.squeeze(0), c_left.squeeze(0)
            sample, h_left, c_left = self.traverse_tree(node.left, h_left, c_left, sample)

        right_logits = self.mlp_right(h)
        right_prob = self.relaxed_bernoulli(right_logits)

        if right_prob > self.right_prob:
            h_right, c_right = self.lstm(h.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
            h_right, c_right = h_right.squeeze(0), c_right.squeeze(0)
            sample, h_right, c_right = self.traverse_tree(node.right, h_right, c_right, sample)

        # Combine the states using BinaryTreeLSTMCell
        if h_left is not None and h_right is not None:
            # Unsqueeze to add the batch dimension
            h_left, c_left = h_left.unsqueeze(0), c_left.unsqueeze(0)
            h_right, c_right = h_right.unsqueeze(0), c_right.unsqueeze(0)

            combined_h, combined_c = self.tree_lstm_cell((h_left, c_left), (h_right, c_right))

            # Squeeze to remove the batch dimension after combining
            combined_h, combined_c = combined_h.squeeze(0), combined_c.squeeze(0)
        elif h_left is not None:
            combined_h, combined_c = h_left, c_left
        elif h_right is not None:
            combined_h, combined_c = h_right, c_right
        else:
            combined_h, combined_c = h, c  # Fall back to the original state if no children are traversed

        return sample, combined_h, combined_c

    def forward(self, h):
        root = self.build_tree(0, self.n - 1)
        c = torch.zeros_like(h)
        row_sample, g_new, c_new = self.traverse_tree(root, h, c)
        return row_sample, g_new
