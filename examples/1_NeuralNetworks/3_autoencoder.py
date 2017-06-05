import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

# Loading the MNIST data set
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])
mnist = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)

# Loader to feed the data batch by batch during training.
batch = 100
data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch, shuffle=True)


autoencoder = nn.Sequential(
                # Encoder
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
    
                # Low-dimensional representation
                nn.Linear(512, 128),
                nn.ReLU(),
    
                # Decoder
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 28 * 28))

autoencoder = autoencoder.type(FloatTensor)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=0.001)

epochs = 10
data_size = int(mnist.train_labels.size()[0])

for i in range(epochs):
    for j, (images, _) in enumerate(data_loader):
        # map tensor from (batch, 1, 28, 28) to (batch, 28 * 28)
        images = images.view(images.size(0), -1)
        images = Variable(images).type(FloatTensor)

        autoencoder.zero_grad()
        reconstructions = autoencoder(images)
        loss = torch.dist(images, reconstructions)
        loss.backward()
        optimizer.step()
    print('Epoch %i/%i loss %.2f' % (i + 1, epochs, loss.data[0]))

print('Optimization finished.')
