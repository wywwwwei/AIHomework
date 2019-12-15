import numpy as np
from cifar import cifarData
import random
import time


class LSVM_Classifier:
    def __init__(self):
        super().__init__()

    '''
    Calculate loss and gradients in vector form
    '''

    def calc_loss(self, XValidate, YValidate, delta=1.0, reg_lambda=0.5):
        num_train = XValidate.shape[0]
        # calcute score of the all classes
        score_all_class = np.dot(XValidate, self.W)
        score_correct_class = score_all_class[
            range(num_train), YValidate]

        # The correct class for each input should  have a score higher than the incorrect classes by some fixed margin delta
        # Hingle loss
        loss_i = np.maximum(0, score_all_class -
                            score_correct_class[:, np.newaxis]+delta)
        loss_i[range(num_train), YValidate] = 0.0

        # Calculate the final loss function (in fact, it can be ignored, because what we need is the gradient)
        data_loss = np.sum(loss_i) / num_train
        regularizer_loss = 0.5 * reg_lambda * np.sum(self.W * self.W)

        # Records the number of classes whose difference from the correct class is greater than margin
        loss_sign = np.zeros(loss_i.shape)
        loss_sign[loss_i > 0] = 1

        count = np.sum(loss_sign, axis=1)
        loss_sign[range(num_train), YValidate] = -count

        deltaW = (np.dot(XValidate.T, loss_sign) /
                  num_train) + reg_lambda * self.W

        return deltaW

    '''
    Adjust the hyperparameters, gradient descent learning rate and regularization strength
    '''

    def hyperparameter_tuning(self, k_fold_num=5):
        learning_rates = [1e-8, 1e-7, 2e-7]
        regularization_strengths = [
            1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 1e5]

        XTrain_fold = np.array_split(self.XTrain, k_fold_num)
        YTrain_fold = np.array_split(self.YTrain, k_fold_num)

        results = np.zeros(
            (len(learning_rates), len(regularization_strengths)))
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                for i in range(k_fold_num):
                    XTrain_cross = np.vstack(XTrain_fold[:i]+XTrain_fold[i+1:])
                    XTest_cross = XTrain_fold[i]

                    YTrain_cross = np.hstack(
                        YTrain_fold[:i] + YTrain_fold[i+1:])
                    YTest_cross = YTrain_fold[i]
                    self.train(XTrain_cross, YTrain_cross, False,
                               learning_rate, regularization_strength)
                    results[learning_rates.index(learning_rate), regularization_strengths.index(
                        regularization_strength)] += self.score(XTest_cross, YTest_cross)

        pos = np.unravel_index(np.argmax(results), results.shape)
        return learning_rates[pos[0]], regularization_strengths[pos[1]]

    '''
    Because we need to iterate many times, it is faster to use stochastic gradient descent
    '''

    def calc_gradient(self, learning_rate, regularization_strength, iteration_num=1000, sample_size=200):
        num_train = self.XTrain.shape[0]
        for i in range(iteration_num):
            sample_index = np.random.choice(
                range(num_train), sample_size, replace=False)
            XValidate = self.XTrain[sample_index, :]
            YValidate = self.YTrain[sample_index]

            grad = self.calc_loss(XValidate, YValidate,
                                  reg_lambda=regularization_strength)
            self.W += -learning_rate * grad

    '''
    Choose whether to adjust hyperparameters through cross-validation according to the parameters
    and then train
    '''

    def train(self, XTrain, YTrain, if_cross_validate=True, learning_rate=1e-7, regularization_strength=1e4):
        self.XTrain = XTrain
        self.YTrain = YTrain
        self.W = 0.001 * np.random.randn(XTrain.shape[1], np.max(YTrain)+1)
        if(if_cross_validate):
            learning_rate, regularization_strength = self.hyperparameter_tuning()
            self.XTrain = XTrain
            self.YTrain = YTrain
            self.calc_gradient(learning_rate, regularization_strength)
        else:
            self.calc_gradient(learning_rate, regularization_strength)

    '''
    Make predictions based on the trained W vectors
    '''

    def predict(self, XTest):
        YPredict = np.zeros(XTest.shape[0])

        score = np.dot(XTest, self.W)
        YPredict = np.argmax(score, axis=1)

        return YPredict

    def score(self, XTest, YTest):
        return np.mean(self.predict(XTest) == YTest)


if __name__ == "__main__":
    train_num = 5000
    test_num = 100
    XTrain, YTrain, XTest, YTest = cifarData().loadCifarData()

    mask = range(XTrain.shape[0])
    mask = random.sample(mask, train_num)
    XTrain = XTrain[mask]
    YTrain = YTrain[mask]
    mask = range(XTest.shape[0])
    mask = random.sample(mask, test_num)
    XTest = XTest[mask]
    YTest = YTest[mask]

    cifarData().normalize_data(XTrain, XTest)
    XTrain = np.hstack([XTrain, np.ones((XTrain.shape[0], 1))])
    XTest = np.hstack([XTest, np.ones((XTest.shape[0], 1))])

    classifier = LSVM_Classifier()
    begintime = time.time()
    classifier.train(XTrain, YTrain, False)
    endtime = time.time()
    print("Train time(no cross validation):%d" % (endtime-begintime,))
    print("%f" % (classifier.score(XTest, YTest),))

    print("after hyperparameter tuning:")
    begintime = time.time()
    classifier.train(XTrain, YTrain)
    endtime = time.time()
    print("Train time(with cross validation):%d" % (endtime-begintime,))
    print("%f" % (classifier.score(XTest, YTest),))
