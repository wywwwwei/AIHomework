import draw
import numpy as np
from typing import List, Tuple, Callable


class ApproximateIntegration:
    def __init__(self, low_bound: float, high_bound: float, f: Callable[[float], float]):
        """Initialize a class that uses Monte Carlo simulation to integrate a function."""
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.f = f

    def generate_integration(self, each_sample: int) -> float:
        """Generate an approximate value of integration.

        Args:
            each_sample: Total number of points thrown.

        Returns:
            approximating integration

        """
        x_rand = np.random.uniform(low=0, high=1.0, size=each_sample)
        x_rand = [self.f(i) for i in x_rand]

        result: float = round((self.high_bound-self.low_bound)
                              * np.sum(x_rand)/each_sample, 5)
        return result

    def generate_repeat(self, each_sample: int, repeat_count: int) -> Tuple[List[float], float, float]:
        """Generate mean and variance.

        Args:
            each_sample: Total number of points thrown.
            repeat_count: The number of repeated throwing process

        Returns:
            all approximate integration, those mean and those variance

        """
        integration_list: List[float] = []

        for i in range(repeat_count):
            integration_list.append(self.generate_integration(each_sample))

        mean: float = round(np.mean(integration_list), 5)
        variance: float = round(np.var(integration_list), 5)

        return integration_list, mean, variance


if __name__ == "__main__":
    approximate_integration = ApproximateIntegration(0, 1.0, lambda x: x ** 3)
    test_case: List[int] = (5, 10, 20, 30, 40, 50, 60, 70, 80, 100)

    means: List[float] = []
    variances: List[float] = []

    for each_sample in test_case:
        cur_result: Tuple[List[float], float, float] = approximate_integration.generate_repeat(
            each_sample=each_sample, repeat_count=100)
        means.append(cur_result[1])
        variances.append(cur_result[2])
        print('N = {} : {}'.format(each_sample, cur_result[0]))

    draw.Draw().draw_mean_variance(abscissa=test_case, mean=means, variance=variances)
