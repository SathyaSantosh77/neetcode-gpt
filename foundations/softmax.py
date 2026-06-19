import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        ns = np.exp(z-max(z))
        smax = ns / np.sum(ns)
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        return np.round(smax, 4)
