from typing import List
import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        super().__init__()

    def draw_loss(self, variable_x: List[int], loss: List[float], ifSGD: bool = False):
        """Draw two pictures, the abscissa is the number of epoches, the ordinate is |error| and loss respectively

        Args:
            variable_x: list of epoch
            error: list of corresponding |error|
            loss: list of corresponding loss
        """
        fig, ax = plt.subplots()
        ax.plot(variable_x, loss, '.-')

        if(ifSGD):
            ax.set(xlabel='iteration(s)',
                   title='Logistic Regression Loss Using SGD')
        else:
            ax.set(xlabel='epoch(es)', title='Logistic Regression Loss')

        ax.set(ylabel='loss')
        plt.show()

    def draw_score(self, variable_x: List[int], score: List[float], ifSGD: bool = False):
        """Draw a pictures, the abscissa is the number of epoches, the ordinate is score

        Args:
            variable_x: list of epoches
            score: list of score
            ifSGD: Whether the current data comes from the process of stochastic gradient descent
        """
        fig, ax = plt.subplots()
        ax.plot(variable_x, score, '.-')

        if(ifSGD):
            ax.set(xlabel='iteration(s)',
                   title='Logistic Regression Loss Using SGD')
        else:
            ax.set(xlabel='epoch(es)', title='Logistic Regression Loss')

        ax.set(ylabel='score')
        plt.show()

    def draw_loss_batchK(self, batch_size: List[int], train_loss: List[float], test_loss: List[float]):
        """Plotting data from the process of mini-batch gradient descent

        Args:
                batch_size: List of batch_size
                train_loss: List of (mean of) training loss
                test_loss: List of (mean of) testing loss

        """

        plt.plot(batch_size, train_loss, 'b.-', label='train loss')
        plt.plot(batch_size, test_loss, 'r.-', label='test loss')
        plt.xlabel('training set size')
        plt.ylabel('loss')
        plt.title('error change as the training set size increases')

        plt.legend()
        plt.show()
    
    def draw_score_batchK(self, batch_size: List[int], train_score: List[float], test_score: List[float]):
        """Plotting data from the process of mini-batch gradient descent

        Args:
                batch_size: List of batch_size
                train_score: List of (mean of) training score
                test_score: List of (mean of) testing score

        """

        plt.plot(batch_size, train_score, 'b.-', label='train score')
        plt.plot(batch_size, test_score, 'r.-', label='test score')
        plt.xlabel('training set size')
        plt.ylabel('score')
        plt.title('error change as the training set size increases')

        plt.legend()
        plt.show()
