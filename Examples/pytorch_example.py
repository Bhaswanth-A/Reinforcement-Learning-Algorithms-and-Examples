import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


# nn.Module gives us access to parameter of our deep neural network
class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        # super constructor calls the constructor for the base class
        super(LinearClassifier, self).__init__()

        # star is to unpack a list corresponding to the elements of the obs vector
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameter(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

        # Check for gpu cuda
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)  # send full network to gpu/cpu

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        cost.backward()
        self.optimizer.step()
