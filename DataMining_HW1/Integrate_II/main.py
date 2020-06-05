import draw
import numpy as np
from typing import List, Tuple, Callable


class ApproximateIntegration:
    def __init__(self, lower_bound: Tuple[float, float], upper_bound: Tuple[float, float], f: Callable[[float, float], float]):
        """Initialize a class that uses Monte Carlo simulation to integrate a 2d function."""
        super().__init__()
        self.volume = np.abs(
            lower_bound[0] - upper_bound[0]) * np.abs(lower_bound[1] - upper_bound[1])
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.f = f

    def generate_integration(self, each_sample: int) -> float:
        """Generate an approximate value of integration.

        Args:
            each_sample: Total number of points thrown.

        Returns:
            approximating integration

        """
        x_rand = np.random.uniform(
            low=self.lower_bound[0], high=self.upper_bound[0], size=each_sample)
        y_rand = np.random.uniform(
            low=self.lower_bound[1], high=self.upper_bound[1], size=each_sample)
        point_rand = np.column_stack((x_rand, y_rand))
        point_rand = [self.f(i[0], i[1]) for i in point_rand]

        result: float = round(self.volume
                              * np.sum(point_rand)/each_sample, 5)
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
    approximate_integration = ApproximateIntegration((2.0, -1.0), (4.0, 1.0), lambda x, y: (
        y**2*np.exp(-y**2)+x**4*np.exp(-x**2))/(x*np.exp(-x**2)))
    test_case: List[int] = (5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200)

    means: List[float] = []
    variances: List[float] = []

    for each_sample in test_case:
        cur_result: Tuple[List[float], float, float] = approximate_integration.generate_repeat(
            each_sample=each_sample, repeat_count=100)
        means.append(cur_result[1])
        variances.append(cur_result[2])
        print('N = {} : {}'.format(each_sample, cur_result[0]))

    draw.Draw().draw_mean_variance(abscissa=test_case, mean=means, variance=variances)
