class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        x = init
        for i in range(iterations):
            grad = 2 * x
            x = x - learning_rate * grad
        return round(x, 5)
