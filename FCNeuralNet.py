import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt

# Initiating data set of hand-drawn numbers from MNIST
b = 10
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=b, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=b, shuffle=True)


##################################


class Net(nn.Module):  # Define class for fully connected neural net
    def __init__(self):
        super().__init__()  # This inherits the init method from nn.Module (parent class of Net)
        self.fc1 = nn.Linear(28*28, 64)  # These are the fully connected layers
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # This defines the activation of the neurons using the rectified linear function.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


net = Net()  # Net()(y) works as well

optimizer = optim.Adam(net.parameters(), lr=1e-3)  # This is a usage of the Adam optimization algorithm with a
# learning rate of lr = 1e-3 = 0.001

EPOCHS = 3  # number of passes through data

for epoch in range(EPOCHS):  # note that 'epoch' and 'data' are simply just iteration variables
    for data in trainset:
        X, y = data
        net.zero_grad()  # This resets the gradient since we are using batches and not on a low memory machine. Low
        # memory machines should not include this and individually pass batches into net().
        output = net(X.view(-1, 28 * 28))
        loss = F.nll_loss(output, y)  # Since results are scalars we use nll_loss but if our result was a vector
        # other loss functions such as mean square error would be used.
        loss.backward()
        optimizer.step()
    # print(loss)
########################################

# The neural net is now trained. Now we can test the neural net by running the test set (testset) into the net and
# testing the accuracy of the outputs

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:  # Tests if the output of the FCNN is the same as the true value.
                correct += 1
            total += 1

print("Accuracy: ", 100 * round(correct / total, 3), "%")


i = 5  # i can range from 1 to b (b being the batch size)
plt.imshow(X[i].view(28, 28))  # view used because X[i] is a tensor of dimension (1,28*28) so we need to turn this
# back into a 28X28 grid in order for imshow to function properly
print(torch.argmax(net(X[i].view(-1, 28*28))[0]))  # this is the predicted value of the drawn digit
plt.show()
