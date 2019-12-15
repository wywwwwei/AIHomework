import numpy as np
from cifar import cifarData
import time
import random


class kNN_Classifier:
    def train(self, XTrain, YTrain):
        self.XTrain = XTrain
        self.YTrain = YTrain

    def predict(self, XTest, k=1):
        distance = self.compute_distance(XTest)
        num_test = XTest.shape[0]
        YPredict = np.zeros(num_test)

        for i in range(num_test):
            classCount = self.YTrain[np.argsort(distance[i])[:k]]
            YPredict[i] = np.argmax(np.bincount(classCount))

        return YPredict

    def score(self, XTest, YTest, k=1):
        return np.mean(self.predict(XTest, k) == YTest)

    def compute_distance(self, XTest):
        num_train = self.XTrain.shape[0]
        num_test = XTest.shape[0]

        distance = np.zeros((num_test, num_train))

        square_XTrain = np.matrix(np.square(self.XTrain).sum(axis=1))
        square_XTest = np.matrix(np.square(XTest).sum(axis=1))
        begintime = time.time()
        distance = np.sqrt(np.add(np.multiply(
            np.dot(XTest, self.XTrain.T), -2), np.add(square_XTrain, square_XTest.T))).getA()
        endtime = time.time()
        print("runtime", endtime-begintime)
        del square_XTrain, square_XTest
        return distance

    def cross_validation(self, k_fold_num, k_test_range=100):
        XTrain_fold = np.array_split(self.XTrain, k_fold_num)
        YTrain_fold = np.array_split(self.YTrain, k_fold_num)

        k_test_result = []
        for k_test in range(1, k_test_range+1):
            k_test_result.append([])
            for i in range(k_fold_num):
                XTrain_cross = np.vstack(XTrain_fold[:i]+XTrain_fold[i+1:])
                XTest_cross = XTrain_fold[i]

                YTrain_cross = np.hstack(YTrain_fold[:i] + YTrain_fold[i+1:])
                YTest_cross = YTrain_fold[i]

                self.train(XTrain_cross, YTrain_cross)
                k_test_result[k_test-1].append(self.score(
                    XTest_cross, YTest_cross, k_test))

        k_test_result = np.sum(np.reshape(
            np.array(k_test_result), (k_test_range, -1)), axis=1)

        print(k_test_result)
        best_k = np.argmax(k_test_result)+1
        del k_test_result

        return best_k


if __name__ == "__main__":
    train_num = 5000
    test_num = 500
    XTrain, YTrain, XTest, YTest = cifarData().loadCifarData()

    mask = list(range(XTrain.shape[0]))
    mask = random.sample(mask, train_num)
    XTrain = XTrain[mask]
    YTrain = YTrain[mask]
    mask = list(range(XTest.shape[0]))
    mask = random.sample(mask, test_num)
    XTest = XTest[mask]
    YTest = YTest[mask]
    print(YTest.shape)

    classifier = kNN_Classifier()
    classifier.train(XTrain, YTrain)
    print(classifier.score(XTest, YTest, 1))

    print("\nafter cross_validation:")

    best_k = classifier.cross_validation(5, 10)
    print("best k = ", best_k)
    classifier.train(XTrain, YTrain)
    print(classifier.score(XTest, YTest, best_k))
