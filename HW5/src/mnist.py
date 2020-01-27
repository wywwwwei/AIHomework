import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class CNNs_MNIST(nn.Module):
    def __init__(self, epoch=1, lr=1e-3):
        super(CNNs_MNIST, self).__init__()

        self.epoch = epoch
        self.lr = lr

        # Input:(1,28,28)
        # First convolution layer and pooling layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        # (16,28,28)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # (16,14,14)

        # Second convolution layer and pooling layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        # (32,14,14)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # (32,7,7)

        # First fully-connected layer
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)
        # (10,1)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        # Flatten and turn them into a single vector that can be an input for the next stage
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train(model, train_loader, test_loader):
    # Instantiate a SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=model.lr)
    # User Cross-Entropy Loss Function.
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(model.epoch):
        for (XTrain, YTrain) in train_loader:
            # Clear gradients
            optimizer.zero_grad()
            # Forward Propagation
            output = model(XTrain)
            # Calculate Loss: softmax cross-entropy loss
            loss = loss_func(output, YTrain)
            # Back Propagation
            loss.backward()
            # Update parameter
            optimizer.step()

        # Calculate Accuracy
        correct = 0
        total = 0
        for (XTest, YTest) in test_loader:
            output = model(XTest)
            _, YPred = torch.max(output.data, 1)

            total += YTest.size(0)
            correct += (YPred == YTest).sum()

        print("Epoch: {} Loss: {} Accuracy: {}" .format(
            epoch+1, loss.item(), 100*correct/total))


def load_mnist(batch_size=200, file_path="./data/mnist"):
    train_data = datasets.MNIST(
        root=file_path, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.MNIST(
        root=file_path, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_mnist()

    cnns_mnist = CNNs_MNIST(20)
    train(cnns_mnist, train_loader, test_loader)
