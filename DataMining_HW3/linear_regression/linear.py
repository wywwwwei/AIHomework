from typing import List, Tuple
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float, initial_w: np.ndarray):
        """Initialize a linear regression model and set the initial coefficient and learning rate of gradient descent
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.w = initial_w

    def get_loss_and_gradient(self, sets_x: np.ndarray, predict_y: np.ndarray, actual_y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Calculate loss function and gradient

        Args:
            sets_x: X of the sample set used to calculate the loss
            predict_y: The model's predictions for sets_x
            actual_y: Y of the sample set used to calculate the loss
        
        Returns:
            The value of loss function and gradient
        """
        diff: np.ndarray = predict_y - actual_y

        loss: float = np.mean(np.dot(diff.T, diff)) / 2
        gradient = np.dot(sets_x.T, diff) / (2 * sets_x.shape[0])

        return np.sum(np.abs(diff)), loss, gradient

    def train_gradient_descent(self, epoch: int, epoch_per_round: int, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Use batch gradient descent to (train) optimize model coefficients

        Args:
            epoch: Number of trainings with full test set
            epoch_per_round: The number of rounds that recording the fit of current model for the test set and training set
            train_x, train_y, test_x, test_y: Training set and test set
        
        Returns:
            Record of the loss function and score of the model
        """
        history_error = []
        history_loss = []
        # Homework
        history_test_error = []
        history_test_loss = []

        for i in range(epoch):
            predict_y = self.predict(train_x)
            diff, loss, gradient = self.get_loss_and_gradient(
                train_x, predict_y, train_y)

            if (i+1) % epoch_per_round == 0:
                history_error.append(diff)
                history_loss.append(loss)
                # Homework
                test_diff, test_loss, test_gradient = self.get_loss_and_gradient(test_x,
                                                                                 self.predict(test_x), test_y)
                history_test_error.append(test_diff)
                history_test_loss.append(test_loss)

            self.w -= self.learning_rate * gradient

        return history_error, history_loss, history_test_error, history_test_loss

    def train_stochastic_gradient_descent(self, iteration_num: int, iter_per_round: int, batch_size: int, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Use mini-batch gradient descent to (train) optimize model coefficients
        When batch_size = 1, it is stochastic gradient descent
        Args:
            iteration_num: Number of trainings with one batch_size samples
            epoch_per_round: The number of rounds that recording the fit of current model for the test set and training set
            train_x, train_y, test_x, test_y: Training set and test set
        
        Returns:
            Record of the loss function and error of the model

        """
        history_error = []
        history_loss = []
        # Homework
        history_test_error = []
        history_test_loss = []

        train_num = train_x.shape[0]
        for i in range(iteration_num):
            sample_index = np.random.choice(
                range(train_num), batch_size, replace=False)
            diff, loss, gradient = self.get_loss_and_gradient(
                train_x[sample_index, :], self.predict(train_x[sample_index, :]), train_y[sample_index])

            self.w -= self.learning_rate * gradient

            if (i+1) % iter_per_round == 0:
                history_error.append(diff)
                history_loss.append(loss)
                # Homework
                test_diff, test_loss, test_gradient = self.get_loss_and_gradient(test_x,
                                                                                 self.predict(test_x), test_y)
                history_test_error.append(test_diff)
                history_test_loss.append(test_loss)

        return history_error, history_loss, history_test_error, history_test_loss

    def predict(self, x_to_predict: np.ndarray) -> np.ndarray:
        """Make predictions for a given X
        """
        return np.dot(x_to_predict, self.w)
