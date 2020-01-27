import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class CNNs_CIFAR10(nn.Module):
    def __init__(self, epoch=1, lr=1e-3):
        super(CNNs_CIFAR10, self).__init__()

        self.epoch = epoch
        self.lr = lr

        # Input:(3,32,32)
        # First convolution layer and pooling layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        # (16,32,32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # (16,16,16)

        # Second convolution layer and pooling layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        # (32,16,16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # (32,8,8)

        # First fully-connected layer
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=10)
        # (10,1)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        # Flatten and turn them into a single vector that can be an input for the next stage
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train(model, train_loader, test_loader):
    # Instantiate a SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=model.lr,momentum=0.9)
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


def load_cifar10(batch_size=200, file_path="./data/cifar10"):
    # Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10(
        root=file_path, train=True, transform=train_transformations, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)

    # Define transformations for the test set
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_data = datasets.CIFAR10(
        root=file_path, train=False, transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_cifar10()

    cnns_cifar = CNNs_CIFAR10(20)
    train(cnns_cifar, train_loader, test_loader)
