import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        self.x = np.array(x)
        # w: 1D weight array (same length as x)
        self.w = np.array(w)
        # b: scalar bias
        self.b = b
        z = np.dot(self.x,self.w) + b
        # activation: "sigmoid" or "relu"
        if activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return float(np.round(sig, 5))
        if activation == 'relu':
            relu = max(0, z)
            return float(np.round(relu,  5))
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

