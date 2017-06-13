import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

# Loading the MNIST data set
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
mnist = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)

# Create a loader to feed the data batch by batch during training
batch = 300
data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch, shuffle=True)


# Now, we define the autoencoder
autoencoder = nn.Sequential(
                # Encoder
                nn.Conv2d(1, 32, 3, padding=1), # input (1, 28, 28), output (32, 28, 28)
                nn.PReLU(32),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2), # (32, 14, 14)
                nn.Conv2d(32, 64, 3), # (64, 12, 12)
                nn.PReLU(64),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2), # (64, 6, 6))
                
                # Low-dimensional representation
                nn.Conv2d(64, 8, 3), # (8, 4, 4) -- latent representation
                nn.PReLU(8),
                nn.BatchNorm2d(8),
                
                # Decoder
                nn.Conv2d(8, 64, 3, padding=2), # (64, 6, 6)
                nn.UpsamplingNearest2d(scale_factor=2), # (64, 12, 12)
                nn.Conv2d(64, 32, 3, padding=2), # (32, 14, 14)
                nn.PReLU(32),
                nn.BatchNorm2d(32),
                nn.UpsamplingNearest2d(scale_factor=2), # (32, 28, 28)
                nn.Conv2d(32, 1, 3, padding=1))

autoencoder = autoencoder.type(FloatTensor)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=0.005)

epochs = 10
data_size = int(mnist.train_labels.size()[0])

for i in range(epochs):
    for j, (images, _) in enumerate(data_loader):
        images = Variable(images).type(FloatTensor)

        autoencoder.zero_grad()
        reconstructions = autoencoder(images)
        loss = torch.dist(images, reconstructions)
        loss.backward()
        optimizer.step()
        
    print('Epoch %i/%i loss %.4f' % (i + 1, epochs, loss.data[0]))

print('Optimization finished.')
