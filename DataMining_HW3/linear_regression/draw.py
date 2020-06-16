from typing import List
import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        super().__init__()

    def draw_error_and_loss(self, variable_x: List[int], error: List[float], loss: List[float], ifSGD: bool = False):
        """Draw two pictures, the abscissa is the number of epoches, the ordinate is |error| and loss respectively

        Args:
            variable_x: list of epoches
            error: list of corresponding |error|
            loss: list of corresponding loss
        """
        plt.subplot(2, 1, 1)
        plt.plot(variable_x, error, '.-')
        plt.title('|Error| And Loss')
        plt.ylabel('|Error|')

        plt.subplot(2, 1, 2)
        plt.plot(variable_x, loss, '.-')
        if ifSGD:
            plt.xlabel('iteration(s)')
        else:
            plt.xlabel('epoch(es)')
        plt.ylabel('Loss')

        plt.show()
