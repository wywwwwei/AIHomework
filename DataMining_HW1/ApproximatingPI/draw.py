import matplotlib.pyplot as plt
from typing import List


class Draw:
    def __init__(self):
        super().__init__()

    def draw_mean_variance(self, abscissa: List[int], mean: List[float], variance: List[float]):
        """Draw a point chart and a table according to the given mean and variance.

        Args:
                abscissa: List of test value of N
                each_sample: List of mean of approximating pi whose input are abscissa
                repeat_count: List of variance of approximating pi whose input are abscissa

        """
        fig: plt.Figure.figure
        ax1: plt.Axes.axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        x_label: List[str] = [str(i) for i in abscissa]
        ax1.plot(x_label, mean, label='mean', color='g')
        ax2.plot(x_label, variance, label='variance', color='b')

        ax1.get_xaxis().set_ticks([])
        ax1.set_ylabel('mean', color='g')
        ax2.set_ylabel('variance', color='b')

        col_lable = ['mean', 'variance']
        cell_text: List[List[float]] = [mean, variance]
        create_table = plt.table(
            cellText=cell_text, rowLabels=col_lable, colLabels=x_label, colLoc='center', loc='bottom', bbox=[0, -0.25, 1, 0.23])
        create_table.auto_set_font_size(False)
        create_table.set_fontsize(10.0)

        plt.subplots_adjust(left=0.12, bottom=0.2)
        plt.show()
