from typing import Tuple, List
from logistic import LogisticRegression
from draw import Drawer
import numpy as np


def getInput(file_path: str) -> Tuple[List[Tuple[float, float, float, float, float, float]], List[int]]:
    """Return the read X and Y according to the initialization file path

    Args:
        file_path: The path of the file to be read

    Returns:
        Two lists store X and Y respectively, 
        with a combination of values at each index corresponding to a sample

        example:
        X:[(0.7,0.2,0.9,0.4,0.7,0.9),(1.0,0.1,0.9,0.1,0.7,0.6)]
        Y:[0,0]
        (0.7,0.2,0.9,0.4,0.7,0.9,0) is a sample and so is (1.0,0.1,0.9,0.1,0.7,0.6,0).
    """
    features: List[Tuple[float, float, float, float, float, float]] = []
    categories: List[int] = []
    with open(file_path, 'r') as f:
        for row in f:
            col_1, col_2, col_3, col_4, col_5, col_6, col_7 = row.split()
            features.append((float(col_1), float(col_2), float(
                col_3), float(col_4), float(col_5), float(col_6)))
            categories.append(int(col_7))
    return features, categories


if __name__ == "__main__":
    train_x, train_y = getInput('./dataForTrainingLogistic.txt')
    test_x, test_y = getInput('./dataForTestingLogistic.txt')

    train_x = np.hstack((np.array(train_x), np.ones((len(train_x), 1))))
    test_x = np.hstack((np.array(test_x), np.ones(((len(test_x), 1)))))
    train_y = np.array(train_y).reshape(len(train_y))
    test_y = np.array(test_y).reshape(len(test_y))

    lr = LogisticRegression(learning_rate=0.00015,
                            initial_w=np.zeros(train_x.shape[1]))
    # batch gradient descent
    # history_loss, history_test_loss, history_score,_ = lr.train_gradient_descent(
    #    epoch=150000, epoch_per_round=10000, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    # stochastic gradient descent
    history_loss, history_test_loss, history_score,_ = lr.train_stochastic_gradient_descent(
         iteration_num=500000, iter_per_round=100, batch_size=1, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    print('Coefficient:', lr.w)
    variable_x = range(100, 500001, 100)
    Drawer().draw_score(variable_x, history_score, ifSGD=True)
    Drawer().draw_loss(variable_x, history_loss, ifSGD=True)
    Drawer().draw_loss(variable_x, history_test_loss, ifSGD=True)

    # Error of stochastic gradient descent with different batch_size
    all_train_loss: List[float] = []
    all_test_loss: List[float] = []
    all_train_score: List[float] = []
    all_test_score: List[float] = []
    for i in range(10, 401, 10):
        lr = LogisticRegression(learning_rate=0.00015,
                                initial_w=np.zeros(train_x.shape[1]))
        history_loss, history_test_loss, history_score, history_test_score = lr.train_stochastic_gradient_descent(
            iteration_num=1, iter_per_round=1, batch_size=i, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        print('k: ', i, ' Coefficient:', lr.w)
        all_train_loss.append(np.mean(history_loss))
        all_test_loss.append(np.mean(history_test_loss))
        all_train_score.append(np.mean(history_score))
        all_test_score.append(np.mean(history_test_score))
    Drawer().draw_loss_batchK(range(10, 401, 10), all_train_loss, all_test_loss)
    Drawer().draw_score_batchK(range(10, 401, 10), all_train_score, all_test_score)
