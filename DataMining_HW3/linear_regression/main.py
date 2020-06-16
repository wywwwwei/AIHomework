from typing import Tuple, List
from linear import LinearRegression
from draw import Drawer
import numpy as np


def getInput(file_path: str) -> Tuple[List[Tuple[int, float]], List[float]]:
    """Return the read X and Y according to the initialization file path

    Args:
        file_path: The path of the file to be read

    Returns:
        Two lists store X and Y respectively,
        with a combination of values at each index corresponding to a sample

        example:
        X:[(93,0.78),(104,3.82)]
        Y:[637.07,494.08]
        (93,0.78,637.07) is a sample and so is (104,3.82,494.08).
    """
    area_dis: List[Tuple[int, float]] = []
    price: List[float] = []
    with open(file_path, 'r') as f:
        for row in f:
            col_1, col_2, col_3 = row.split()
            area_dis.append((int(col_1), float(col_2)))
            price.append(float(col_3))
    return area_dis, price


if __name__ == "__main__":
    train_x, train_y = getInput('./dataForTrainingLinear.txt')
    test_x, test_y = getInput('./dataForTestingLinear.txt')

    train_x = np.hstack((np.array(train_x), np.ones((len(train_x), 1))))
    test_x = np.hstack((np.array(test_x), np.ones(((len(test_x), 1)))))
    train_y = np.array(train_y).reshape(len(train_y))
    test_y = np.array(test_y).reshape(len(test_y))

    lr = LinearRegression(learning_rate=0.00015,
                          initial_w=np.zeros(train_x.shape[1]))

    #history_error, history_loss, history_test_error, history_test_loss = lr.train_gradient_descent(
    #    epoch=150000, epoch_per_round=10000, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    history_error, history_loss, history_test_error, history_test_loss = lr.train_stochastic_gradient_descent(
        iteration_num=150000, iter_per_round=10000, batch_size=1, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    variable_x = range(10000, 150001, 10000)
    Drawer().draw_error_and_loss(variable_x, history_error, history_loss)
    Drawer().draw_error_and_loss(
        variable_x, history_test_error, history_test_loss)
