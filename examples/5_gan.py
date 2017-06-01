import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise

# Loading the MNIST data set
batch =  100
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])
mnist = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch, shuffle=True)

# --  Hyperparameters --
epochs = 150

# Discriminator
input_size_d = 28 * 28
hidden_size_d = 256
output_size_d = 1 # probability
learning_rate_d = .0005

# Generator
input_size_g = 128
hidden_size_g = 256
output_size_g = input_size_d # output of generator is a flatten image
learning_rate_g = .0005

discriminator = nn.Sequential(
            nn.Linear(input_size_d, hidden_size_d),
            nn.ReLU(),
            nn.Linear(hidden_size_d, hidden_size_d),
            nn.ReLU(),
            nn.Linear(hidden_size_d, output_size_d),
            nn.Sigmoid())

generator = nn.Sequential(
            nn.Linear(input_size_g, hidden_size_g),
            nn.LeakyReLU(),
            nn.Linear(hidden_size_g, hidden_size_g),
            nn.LeakyReLU(),
            nn.Linear(hidden_size_g, output_size_g),
            nn.Tanh())

discriminator = discriminator.cuda() if cuda else disciminator
generator = generator.cuda() if cuda else generator

criterion = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate_d)
optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=learning_rate_g)

data_size = int(mnist.train_labels.size()[0])
d_x = 0. # average classification probabilities of real data
d_g_z = 0. # average classification probabilities of fake data
loss_d = 0. # average loss of discriminator
loss_g = 0. # average loss of generator

for i in range(epochs):
    for j, (images, _) in enumerate(data_loader):
        # map tensor from (batch, 1, 28, 28) to (batch, 28 * 28)
        images = images.view(images.size(0), -1)
        real_images = Variable(images).cuda() if cuda else Variable(images)
        
        ones = Variable(torch.ones(images.size(0))).cuda() if cuda else Variable(torch.ones(images.size(0)))
        zeros = Variable(torch.zeros(images.size(0))).cuda() if cuda else Variable(torch.zeros(images.size(0)))
        
        # Discriminator step
        discriminator.zero_grad()
        
        noise = Variable(torch.randn(batch, input_size_g)).cuda() if cuda else Variable(torch.randn(batch, input_size_g))
        fake_images = generator(noise) # map input noise to the data space (image)
        discriminator_real = discriminator(real_images)
        discriminator_fake = discriminator(fake_images.detach())
        real_loss_discriminator = criterion(discriminator_real, ones)
        fake_loss_discriminator = criterion(discriminator_fake, zeros)
        
        loss_discriminator = real_loss_discriminator + fake_loss_discriminator
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        #  Generator step
        generator.zero_grad()
        
        noise = Variable(torch.randn(batch, input_size_g)).cuda() if cuda else Variable(torch.randn(batch, input_size_g))
        fake_images = generator(noise) # map input noise to the data space (image)
        discriminator_fake2 = discriminator(fake_images)
        loss_generator = criterion(discriminator_fake2, ones)
        loss_generator.backward()
        optimizer_generator.step()
        
        d_x += torch.sum(discriminator_real).data[0] / data_size
        d_g_z += torch.sum(discriminator_fake).data[0] / data_size
        loss_d += loss_discriminator.data[0] / (data_size / batch)
        loss_g += loss_generator.data[0] / (data_size / batch)
        
    print('Epoch %i/%i, d_loss: %.2f, g_loss: %.2f, D(X)=%.2f D(G(Z))=%.2f' % 
                (i + 1, epochs, loss_d,
                loss_g, d_x, d_g_z))    
    d_x, d_g_z, loss_d, loss_g = 0., 0., 0., 0.    

print('Optimization finished.')
