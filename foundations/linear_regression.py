import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        dot = np.dot(X, weights)
        # Round to 5 decimal places
        return np.round(dot, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        n = len(ground_truth)
        mse = 1/n * np.sum(np.pow((model_prediction - ground_truth),2))
        # Round to 5 decimal places
        return np.round(mse, 5)
