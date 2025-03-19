import torch.nn as nn


class ReLU2(nn.Module):
    """ReLU2 activation function.

    Based on the paper 'Primer: Searching for Efficient Transformers for Language Modeling'.
    Link: https://arxiv.org/abs/2109.08668
    """

    def __init__(self):
        """Initializes the ReLU2 module."""
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Computes the output of the Relu2 module.

        This function takes as input a tensor `x` and returns the element-wise
        square of the ReLU activation function of `x`.
        """
        return self.relu(x) ** 2
