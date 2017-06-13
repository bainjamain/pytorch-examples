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
train_data = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='../data/', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch)

# Now, we define the recurrent neural network
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=1024, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(1024, 10),
                    nn.Softmax())
    
    def forward(self, x):
        h0 = Variable(torch.randn(1, x.size(0), 1024)).type(FloatTensor)
        c0 = Variable(torch.randn(1, x.size(0), 1024)).type(FloatTensor)
        x, _  = self.rnn(x, (h0, c0))
        x = x[:, -1, :] # last output
        return self.fc(x)

rnn = RNN().type(FloatTensor)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=0.001)

epochs = 5
train_size = int(train_data.train_labels.size()[0])
test_size = int(test_data.test_labels.size()[0])
accuracy = 0.

for i in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = Variable(images).view(images.size(0), 28, 28)
        images = images.type(FloatTensor)
        labels = Variable(labels).type(LongTensor)

        rnn.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # test network  
        if (j + 1) % 300 == 0:
            for images, labels in test_loader:
                images = Variable(images).view(images.size(0), 28, 28).type(FloatTensor)
                labels = Variable(labels).type(LongTensor)
                outputs = rnn(images)
                _, predicted = torch.max(outputs, 1)
                accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
            print('[TEST] Epoch %i/%i [step %i/%i] accuracy: %.3f' % 
                  (i + 1, epochs, j + 1, float(train_size) / batch, accuracy))
            accuracy = 0.

print('Optimization finished.')
