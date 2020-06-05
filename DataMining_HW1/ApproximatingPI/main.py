import random
import draw
import numpy as np
from typing import List, Tuple


class ApproximatePI:
    def __init__(self):
        """Initialize a class that uses Monte Carlo simulation to approximate pi."""
        super().__init__()

    def generate_pi(self, each_sample: int) -> float:
        """Generate an approximate value of pi.

        Args:
            each_sample: Total number of points thrown.

        Returns:
            approximating pi

        """
        x_rand = np.random.uniform(low=0, high=1.0, size=each_sample)
        y_rand = np.random.uniform(low=0, high=1.0, size=each_sample)

        result: float = round(np.mean(x_rand ** 2 + y_rand ** 2 <= 1) * 4, 5)
        return result

    def generate_repeat(self, each_sample: int, repeat_count: int) -> Tuple[List[float], float, float]:
        """Generate mean and variance.

        Args:
            each_sample: Total number of points thrown.
            repeat_count: The number of repeated throwing process

        Returns:
            all approximate pi, those mean and those variance

        """
        pi_list: List[float] = []

        for i in range(repeat_count):
            pi_list.append(self.generate_pi(each_sample))

        mean: float = round(np.mean(pi_list), 5)
        variance: float = round(np.var(pi_list), 5)

        return pi_list, mean, variance


if __name__ == "__main__":
    approximate_pi = ApproximatePI()
    test_case: List[int] = (50, 100, 200, 300, 500, 1000, 5000)

    means: List[float] = []
    variances: List[float] = []

    for each_sample in test_case:
        cur_result: Tuple[List[float], float, float] = approximate_pi.generate_repeat(
            each_sample=each_sample, repeat_count=20)
        means.append(cur_result[1])
        variances.append(cur_result[2])
        print('N = {} : {}'.format(each_sample, cur_result[0]))

    draw.Draw().draw_mean_variance(abscissa=test_case, mean=means, variance=variances)
