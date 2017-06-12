import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

# Loading the MNIST data set
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])
train_data = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='../data/', train=False, transform=transform, download=True)

# Loader to feed the data batch by batch during training.
batch = 100
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch)

net = torch.nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10))
net = net.type(FloatTensor)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

epochs = 5
train_size = int(train_data.train_labels.size()[0])
test_size = int(test_data.test_labels.size()[0])
accuracy = 0.

for i in range(epochs):
    # train network
    for j, (images, labels) in enumerate(train_loader):
        # map tensor from (batch, 1, 28, 28) to (batch, 28 * 28)
        images = Variable(images.view(batch, -1)).type(FloatTensor)
        labels = Variable(labels).type(LongTensor)

        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
    # test network
    for images, labels in test_loader:
        images = Variable(images.view(batch, -1)).type(FloatTensor)
        labels = Variable(labels).type(LongTensor)
        outputs = net(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
    print('[TEST] Epoch %i/%i loss: %.2f, accuracy: %.3f' % (i + 1, epochs, loss.data[0], accuracy))
    accuracy = 0.

print('Optimization finished.')
