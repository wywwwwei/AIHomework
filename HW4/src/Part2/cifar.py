import pickle
import numpy as np
import os


class cifarData:
    def __init__(self, root="./batches"):
        self.root = root

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding="bytes")
            X = dict[b'data']
            Y = dict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(
                0, 2, 3, 1).astype("float")
            Y = np.array(Y)
        return X, Y

    def loadCifarData(self):
        xdata = []
        ydata = []
        for b in range(1, 6):
            filename = os.path.join(self.root, 'data_batch_%d' % (b))
            X, Y = self.unpickle(filename)
            xdata.append(X)
            ydata.append(Y)
        XTrain = np.concatenate(xdata)
        YTrain = np.concatenate(ydata)
        del xdata, ydata
        XTest, YTest = self.unpickle(os.path.join(self.root, 'test_batch'))

        XTrain = np.reshape(XTrain, (XTrain.shape[0], 32*32*3))
        XTest = np.reshape(XTest, (XTest.shape[0], 32*32*3))
        YTrain = np.reshape(YTrain, len(YTrain))
        YTest = np.reshape(YTest, len(YTest))

        return XTrain, YTrain, XTest, YTest

    def normalize_data(self, XTrain, XTest):
        mean_x = np.mean(XTrain, axis=0)
        XTrain -= mean_x
        XTest -= mean_x
        return XTrain
