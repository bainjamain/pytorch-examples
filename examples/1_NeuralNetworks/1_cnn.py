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

# Create a loader to feed the data batch by batch during training
batch = 100
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch)

# Define the convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # input (1, 28, 28), output (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 14, 14)
            nn.Conv2d(32, 64, 3), # (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)) # (64, 6, 6))
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(512, 10),
            nn.Softmax())
    
    def forward(self, x):
        x  = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
cnn = CNN().type(FloatTensor)

criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=0.001)

epochs = 5
train_size = int(train_data.train_labels.size()[0])
test_size = int(test_data.test_labels.size()[0])
accuracy = 0.

for i in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        cnn.train()
        images = Variable(images).type(FloatTensor)
        labels = Variable(labels).type(LongTensor)

        cnn.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # test network  
        if (j + 1) % 300 == 0:
            cnn.eval()
            for images, labels in test_loader:
                images = Variable(images).type(FloatTensor)
                labels = Variable(labels).type(LongTensor)
                outputs = cnn(images)
                _, predicted = torch.max(outputs, 1)
                accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
            print('[TEST] Epoch %i/%i [step %i/%i] accuracy: %.3f' % 
                  (i + 1, epochs, j + 1, float(train_size) / batch, accuracy))
            accuracy = 0.

print('Optimization finished.')
