import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

# Loading the MNIST data set
train_data = torchvision.datasets.MNIST(root='../data/', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='../data/', train=False, download=True)

# Create pairs of similar and dissimilar images
def make_pairs(data, labels, num=1000):
    digits = {}
    for i, j in enumerate(labels):
        if not j in digits:
            digits[j] = []
        digits[j].append(i)

    pairs, labels_ = [], []
    for i in range(num):
        if np.random.rand() >= .5: # same digit
            digit = np.random.choice(range(10))
            d1, d2 = np.random.choice(digits[digit], size=2, replace=False)
            labels_.append(1)
        else:
            digit1, digit2 = np.random.choice(range(10), size=2, replace=False)
            d1, d2 = np.random.choice(digits[digit1]), np.random.choice(digits[digit2])
            labels_.append(0)
        pairs.append(torch.cat([data[d1], data[d2]]).view(1, 56, 28))
    return torch.cat(pairs), torch.LongTensor(labels_)

batch = 100
pairs_train, labels_train = make_pairs(train_data.train_data, train_data.train_labels, num=60000)
pairs_test, labels_test = make_pairs(test_data.test_data, test_data.test_labels, num=10000)
train = torch.utils.data.dataset.TensorDataset(pairs_train, labels_train)
test = torch.utils.data.dataset.TensorDataset(pairs_test, labels_test)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch)

net = torch.nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128))

bottom_net = torch.nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 2))

net = net.type(FloatTensor)
bottom_net = bottom_net.type(FloatTensor)

params = [x for x in net.parameters()] + [x for x in bottom_net.parameters()]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=params, lr=0.001)

epochs = 5
train_size = int(labels_train.size()[0])
test_size = int(labels_test.size()[0])
accuracy = 0.
for i in range(epochs):
    # train network
    for j, (images, labels) in enumerate(train_loader):
        i1, i2 = images.view(batch, -1).split(28 * 28, dim=1)
        i1 = Variable(i1.type(FloatTensor))
        i2 = Variable(i2.type(FloatTensor))
        labels = Variable(labels).type(LongTensor)

        net.zero_grad()
        bottom_net.zero_grad()
        output1 = net(i1)
        output2 = net(i2)
        output = bottom_net((output1 - output2) ** 2)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
            
        # test network
        if (j + 1) % 600 == 0:
            for k, (images, labels) in enumerate(test_loader):
                i1, i2 = images.view(batch, -1).split(28 * 28, dim=1)
                i1 = Variable(i1.type(FloatTensor))
                i2 = Variable(i2.type(FloatTensor))
                labels = Variable(labels).type(LongTensor)
                output1 = net(i1)
                output2 = net(i2)
                output = bottom_net((output1 - output2) ** 2)
                _, predicted = torch.max(output, 1)
                accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
            print('[TEST] Epoch %i/%i [step %i/%i] accuracy: %.3f' % (i + 1, epochs, j + 1, train_size / batch, accuracy))
            accuracy = 0.

print('Optimization finished.')
