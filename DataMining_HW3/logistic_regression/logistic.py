from typing import Tuple, List
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate: float, initial_w: np.ndarray):
        """
        Initialize a logistic regression model and set the initial coefficient and learning rate of gradient descent
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.w = initial_w

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """A function to map variables to between 0 and 1
        """
        return 1/(1+np.exp(-x))

    def get_loss_and_gradient(self, sets_x: np.ndarray, predict_y: np.ndarray, actual_y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate loss function and gradient

        Args:
            sets_x: X of the sample set used to calculate the loss
            predict_y: The model's predictions for sets_x
            actual_y: Y of the sample set used to calculate the loss
        
        Returns:
            The value of loss function and gradient
        """
        case_num = sets_x.shape[0]

        loss: float = -(np.dot(actual_y, np.log(self.sigmoid(predict_y))) +
                        np.dot(1-actual_y, np.log(1-self.sigmoid(predict_y))))/case_num
        gradient = np.dot(sets_x.T, actual_y -
                          self.sigmoid(predict_y))/case_num

        return loss, gradient

    def train_gradient_descent(self, epoch: int, epoch_per_round: int, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Use batch gradient descent to (train) optimize model coefficients

        Args:
            epoch: Number of trainings with full test set
            epoch_per_round: The number of rounds that recording the fit of current model for the test set and training set
            train_x, train_y, test_x, test_y: Training set and test set
        
        Returns:
            Record of the loss function and score of the model
        """
        history_loss = []
        # Homework
        history_test_loss = []
        history_score = []
        history_test_score = []

        for i in range(epoch):
            predict_y = self.predict(train_x)
            loss, gradient = self.get_loss_and_gradient(
                train_x, predict_y, train_y)

            if (i+1) % epoch_per_round == 0:
                history_loss.append(loss)
                # Homework
                test_loss, test_gradient = self.get_loss_and_gradient(
                    test_x, self.predict(test_x), test_y)
                history_test_loss.append(test_loss)
                history_score.append(self.score(train_x, train_y))
                history_test_score.append(self.score(test_x, test_y))

            self.w += self.learning_rate * gradient

        return history_loss, history_test_loss, history_score, history_test_score

    def train_stochastic_gradient_descent(self, iteration_num: int, iter_per_round: int, batch_size: int, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Tuple[List[float],List[float], List[float], List[float]]:
        """Use mini-batch gradient descent to (train) optimize model coefficients
        When batch_size = 1, it is stochastic gradient descent
        Args:
            iteration_num: Number of trainings with one batch_size samples
            epoch_per_round: The number of rounds that recording the fit of current model for the test set and training set
            train_x, train_y, test_x, test_y: Training set and test set
        
        Returns:
            Record of the loss function and score of the model

        """
        history_loss = []
        # Homework
        history_test_loss = []
        history_score = []
        history_test_score = []

        train_num = train_x.shape[0]
        for i in range(iteration_num):
            sample_index = np.random.choice(
                range(train_num), batch_size, replace=False)
            loss, gradient = self.get_loss_and_gradient(
                train_x[sample_index, :], self.predict(train_x[sample_index, :]), train_y[sample_index])

            self.w += self.learning_rate * gradient

            if (i+1) % iter_per_round == 0:
                history_loss.append(loss)
                # Homework
                test_loss, test_gradient = self.get_loss_and_gradient(
                    test_x, self.predict(test_x), test_y)
                history_test_loss.append(test_loss)
                history_score.append(self.score(train_x[sample_index, :],train_y[sample_index]))
                history_test_score.append(self.score(test_x, test_y))

        return history_loss, history_test_loss, history_score,history_test_score

    def predict(self, x_to_predict: np.ndarray) -> np.ndarray:
        """Make predictions for a given X
        """
        return np.dot(x_to_predict, self.w)

    def score(self, test_x: np.ndarray, test_y: np.ndarray, threshold: float = 0.5) -> float:
        """Prediction accuracy for the current sample set
        """
        predict_y = (self.sigmoid(self.predict(test_x))
                     >= threshold).astype(int)
        return np.mean(predict_y == test_y)
