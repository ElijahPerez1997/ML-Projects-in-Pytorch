# This program implements a Convolutional Neural Network (CNN) using Pytorch. This CNN learns from images of cats and
# dogs and learns to differentiate between the two. Passing new images to the CNN that were not used to teach the
# network can test the CNN.
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


REBUILD_DATA = True  # This can be switched between true and false to rebuild dataset


class DogsAndCats():  # DogsAndCats() class processes cat and dog images from the PetImages folder and prepares them
    # for entry into the CNN.
    IMG_SIZE = 50  # This is the image size. Images will be reduced to a 50x50 image
    CATS = "PetImages/Cat"  # This is a dir call to cat images from the PetImages folder
    DOGS = "PetImages/Dog"  # This is a dir call to cat images from the PetImages folder
    TESTING = "PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:  # This is to ensure that the filetype is jpg. Some files in the folder may not be.
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Converts image to grayscale to simplify CNN.
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))  # Converts image to 50x50
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # Assigning 1 hot
                        # vectors associated with cats and dogs. [1,0] for cat and [0,1] for dog.

                        if label == self.CATS:  # Keeps track of the number of dog and cat images. These should be
                            # roughly equal so that the CNN is trained with a near equal number of dog and cat images.
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1

                    except Exception as e:
                        pass  # pass here instead of exception handling since some files are corrupted inside the
                        # PetImage folder. Pass used to simply ignore these corrupted files.

        np.random.shuffle(self.training_data)  # This will shuffle the images
        np.save("training_data.npy", self.training_data)  # Saves training data as training_data.npy
        print('Cats:', dogsandcats.catcount)
        print('Dogs:', dogsandcats.dogcount)


if REBUILD_DATA:
    dogsandcats = DogsAndCats()
    dogsandcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)


# print(len(training_data))

###################################
# To test what the images look like, run the following lines:

# import matplotlib.pyplot as plt
# X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# X = X / 255
# y = torch.Tensor([i[1] for i in training_data])
# j = 1  # This j value designates which image you want to show. This can vay from 0 to len(training_data)-1
# plt.imshow(X[j], cmap = "gray")  # This will display the image exactly how the CNN will see it.
# print(y[j])  # Will return [1,0] for cat image and [0,1] for dog image

###################################
class Net(nn.Module):  # This class defines the CNN
    def __init__(self):
        super().__init__()  # inherits __init__ from super (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5)  # These define the conv layers as well as pooling. Conv layers have kernel
        # of 5x5, initially output 32 and output doubles for each conv layer. Input starts with 1 image, then is just
        # the output of previous conv layer.
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.MaxPool2d((2, 2))

        ##############################################
        # To find the value for the input to the first fully connected layer (self.fc1), comment out code with "$$"
        # and run. The output will be a 2d tensor, with torch.Size([1, 512]) hence where input value 512 comes from.
        # This will only need to be done once so that you can find the value of the first input of the fc layer.
        ##############################################

        self.fc1 = nn.Linear(512, 500)  # $$
        self.fc2 = nn.Linear(500, 2)  # $$

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.flatten(start_dim=1)  # flattening out for use in fully connected layers
        #  print(x.shape)  # Uncomment when doing "$$"

        x = F.relu(self.fc1(x))  # $$
        x = self.fc2(x)  # $$
        return F.softmax(x, dim=1)  # $$


net = Net()
# net.forward(torch.randn(1, 1, 50, 50))  # Uncomment when doing "$$"

optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use the Adam optimization algorithm on all free parameters of
# the CNN with a fixed learning rate of lr=0.001
loss_function = nn.MSELoss()  # Use mean square error as loss since outputs are 1 hot vectors

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)  # Tensor X contains the raw data of the image.
X = X / 255  # This will make all entries in the data array (tensor) X between 0 and 1. This is because gray scale
# images in matplotlib are of this form.
y = torch.Tensor([i[1] for i in training_data])  # This is a 1 hot vector that classifies the image. [1,0] for cat
# and [0,1] for dog.

test_percent = 0.1  # Letting 10% of the images from data be reserved for testing the CNN.
test_size = int(len(X) * test_percent)  # Getting the number of images that will be in our test set.

train_X = X[:-test_size]  # Splitting X and y into training and testing sets
train_y = y[:-test_size]
test_X = X[-test_size:]
test_y = y[-test_size:]

BATCH_SIZE = 100  # Defines the size of the batches of data that will be passed to the CNN
EPOCHS = 1  # Defines number of passes through data. This will have a direct impact on accuracy (ie increasing EPOCHS
# should increase accuracy). However, this is a CPU driven CNN and not a GPU driven CNN, so run time will drastically
# increase with EPOCHS size. Note that EPOCHS must be a positive integer.

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + BATCH_SIZE]

        net.zero_grad()  # On low memory machines, this would not be recommended. Low memory machines would
        # individually run batches though and not call net.zero_grad(). This would increase the grad by summing the
        # previous grads, rather than doing this all in one step.

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()  # Computes gradients for loss.
        optimizer.step()  # Updates based on gradient calculation from line above.
    print(f"Epoch: {epoch+1}. Loss: {loss}")

# ##################################################################
# The CNN is now trained. The accuracy of the CNN can now be tested by running the testing data (ie test_X) into the
# CNN and comparing this output to the true value (test_y). By finding the ratio of correct to incorrect predictions,
# we can get a measure of accuracy of the CNN.
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_animal = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list
        predicted_animal = torch.argmax(net_out)

        if predicted_animal == real_animal:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))
