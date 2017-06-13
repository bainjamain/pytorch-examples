import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

# Loading the MNIST data set.
batch = 100
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
mnist = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch, shuffle=True)

z_dim = 3 # dimension of the low-dimensional code / representation

# autoencoder
encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.PReLU(512),
            nn.BatchNorm1d(512),
            nn.Linear(512, z_dim))

decoder = nn.Sequential(
            nn.PReLU(z_dim),
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, 512),
            nn.PReLU(512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid())

autoencoder = nn.Sequential(
            encoder,
            decoder)

# discriminator
discriminator = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.PReLU(512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid())

encoder = encoder.type(FloatTensor)
decoder = decoder.type(FloatTensor)
autoencoder = autoencoder.type(FloatTensor)
discriminator = discriminator.type(FloatTensor)

lr = 0.001 # learning rate
criterion = nn.BCELoss()
optimizer_ae = torch.optim.Adam(params=autoencoder.parameters(), lr=lr)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=lr)
optimizer_g = torch.optim.Adam(params=encoder.parameters(), lr=lr)

epochs = 10
data_size = int(mnist.train_labels.size()[0])

for i in range(epochs):
    for j, (images, _) in enumerate(data_loader):
        images = images.view(batch, -1)
        images = Variable(images).type(FloatTensor)
        
        ones = Variable(torch.ones(images.size(0))).type(FloatTensor)
        zeros = Variable(torch.zeros(images.size(0))).type(FloatTensor)

        # Autoencoder step
        autoencoder.zero_grad()
        reconstructions = autoencoder(images)
        loss_ae = torch.dist(images, reconstructions)
        loss_ae.backward()
        optimizer_ae.step()

        # Discriminator step
        discriminator.zero_grad()
        z_fake = encoder(images.detach())
        z_real = Variable(torch.randn(z_fake.size())).type(FloatTensor)
        z_fake_d = discriminator(z_fake)
        z_real_d = discriminator(z_real)

        loss_real = criterion(z_real_d, ones)
        loss_fake = criterion(z_fake_d, zeros)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Generator (encoder) step
        encoder.zero_grad()
        discriminator
        z_fake = encoder(images)
        z_fake_d = discriminator(z_fake)
        loss_g = criterion(z_fake_d, ones)
        loss_g.backward()
        optimizer_g.step()
        if (j + 1) % 100 == 0:
            print('Epoch %i/%i [%i/%i] loss_ae: %.2f, D(z_real)=%.2f, D(E(x))=%.2f' % (i + 1, epochs, j + 1,
                            data_size / float(batch), loss_ae.data[0], z_real_d.mean().data[0],
                            z_fake_d.mean().data[0]))

print('Optimization finished.')
