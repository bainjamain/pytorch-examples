import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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

# Loader to feed the data batch by batch during training
batch = 100
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch)

# Defining the fully convolutional neural network
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # input (1, 28, 28), output (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3, padding=1), # (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (128, 14, 14))
            
            nn.Conv2d(128, 128, 3, padding=1), # (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), # (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (128, 7, 7)
            nn.Conv2d(128, 10, 1)) # (10, 7, 7)
            
    def forward(self, x):
        x  = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        return F.softmax(x)
    
fcnn = FCNN().type(FloatTensor)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=fcnn.parameters(), lr=0.001)

epochs = 5
train_size = int(train_data.train_labels.size()[0])
test_size = int(test_data.test_labels.size()[0])
accuracy = 0.

for i in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        fcnn.train()
        images = Variable(images).type(FloatTensor)
        labels = Variable(labels).type(LongTensor)

        fcnn.zero_grad()
        outputs = fcnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # test network  
        if (j + 1) % 300 == 0:
            fcnn.eval()
            for images, labels in test_loader:
                images = Variable(images).type(FloatTensor)
                labels = Variable(labels).type(LongTensor)
                outputs = fcnn(images)
                _, predicted = torch.max(outputs, 1)
                accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
            print('[TEST] Epoch %i/%i [step %i/%i] accuracy: %.3f' % 
                  (i + 1, epochs, j + 1, float(train_size) / batch, accuracy))
            accuracy = 0.

print('Optimization finished.')
