import numpy as np
from cifar import cifarData


class ANN_Classifier:
    def __init__(self, input_dimension, hidden_dimension, output_dimension, std=1e-4):
        self.params = {
            "W1": std * np.random.randn(input_dimension, hidden_dimension), "b1": np.zeros(hidden_dimension),
            "W2": std * np.random.randn(hidden_dimension, output_dimension), "b2": np.zeros(output_dimension)}

    def calc_loss_and_gradients(self, XValidate, YValidate):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        validate_size = XValidate.shape[0]
        # Forward pass
        # First layer pre-activation
        score_layer_first = np.add(np.dot(XValidate, W1), b1)

        # First layer activation (ReLU)
        score_ReLU = np.maximum(0, score_layer_first)  # 0 broadcast dimension

        # Second layer pre-activation
        score_layer_second = np.add(np.dot(score_ReLU, W2), b2)

        # Second layer activation (Softmax)
        score_Sofmax = np.exp(
            score_layer_second) / np.sum(np.exp(score_layer_second), axis=1, keepdims=True)

        # cross-entropy loss
        data_loss = np.sum(
            -np.log(score_Sofmax[range(validate_size), YValidate])) / validate_size
        reg_loss = 0.5 * self.regularization_strength * \
            np.sum(W1 * W1) + 0.5 * \
            self.regularization_strength * np.sum(W2 * W2)
        loss = data_loss + reg_loss

        gradients = {}
        # Backward Pass
        d_scores = score_Sofmax
        d_scores[range(validate_size), YValidate] -= 1
        d_scores /= validate_size

        d_W2 = np.dot(score_ReLU.T, d_scores)
        d_b2 = np.sum(d_scores, axis=0)

        # Propagate to second layer
        d_layer_second = np.dot(d_scores, W2.T)

        # Backprop the ReLU non-linearity
        d_layer_second[score_ReLU <= 0] = 0

        d_W1 = np.dot(XValidate.T, d_layer_second)
        d_b1 = np.sum(d_layer_second, axis=0)

        gradients['W2'] = d_W2 + self.regularization_strength * W2
        gradients['b2'] = d_b2
        gradients['W1'] = d_W1 + self.regularization_strength * W1
        gradients['b1'] = d_b1

        return loss, gradients

    def gradient_descent(self):
        num_train = self.XTrain.shape[0]

        for i in range(self.iteration_num):
            sample_index = np.random.choice(
                range(num_train), self.sample_size, replace=False)
            XValidate = self.XTrain[sample_index, :]
            YValidate = self.YTrain[sample_index]

            loss, gradients = self.calc_loss_and_gradients(
                XValidate, YValidate)

            self.params["W1"] -= self.learnign_rate * gradients["W1"]
            self.params["b1"] -= self.learnign_rate * gradients["b1"]
            self.params["W2"] -= self.learnign_rate * gradients["W2"]
            self.params["b2"] -= self.learnign_rate * gradients["b2"]

    def train(self, XTrain, Ytrain, iteration_num=1000, sample_size=200, learning_rate=1e-3, regularization_strength=0.5):
        self.XTrain = XTrain
        self.YTrain = YTrain
        self.iteration_num = iteration_num
        self.sample_size = sample_size
        self.learnign_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.gradient_descent()

    def predict(self, XPredict):
        result_layer_first = np.add(
            np.dot(XPredict, self.params["W1"]), self.params["b1"])
        result_ReLU = np.maximum(0, result_layer_first)
        result_layer_second = np.add(
            np.dot(result_ReLU, self.params["W2"]), self.params["b2"])
        YPredict = np.argmax(result_layer_second, axis=1)
        return YPredict

    def score(self, XPredict, YPredict):
        return np.mean(YPredict == self.predict(XPredict))


if __name__ == "__main__":
    train_num = 5000
    test_num = 100
    XTrain, YTrain, XTest, YTest = cifarData().loadCifarData()

    cifarData().normalize_data(XTrain, XTest)

    classifier = ANN_Classifier(
        input_dimension=XTrain.shape[1], hidden_dimension=100, output_dimension=10)
    classifier.train(XTrain, YTrain)
    print("accuracy:%f" % (classifier.score(XTest, YTest)))
