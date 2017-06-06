import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import time

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])
train_data = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='../data/', train=False, transform=transform, download=True)

batch = 600 # 2 GPUs, batch of 600/2=300 images per GPU
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(1024, 10),
                    nn.Softmax())
    
    def forward(self, x):
        x, _  = self.rnn(x)
        x = x[:, -1, :] # last output
        return self.fc(x)

rnn = RNN().cuda()
rnn = torch.nn.DataParallel(rnn, device_ids=[0, 1]) # magic line to parallelize on multiple GPUs

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=0.001)

epochs = 5
train_size = int(train_data.train_labels.size()[0])
test_size = int(test_data.test_labels.size()[0])
accuracy = 0.

start = time.time()
for i in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = Variable(images).view(images.size(0), 28, 28)
        labels = Variable(labels).cuda()

        rnn.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # test network  
        if (j + 1) % 100 == 0:
            for images, labels in test_loader:
                images = Variable(images).view(images.size(0), 28, 28)
                labels = Variable(labels).cuda()
                outputs = rnn(images)
                _, predicted = torch.max(outputs, 1)
                accuracy += torch.sum(torch.eq(predicted, labels).float()).data[0] / test_size
            print('[TEST] Epoch %i/%i [step %i/%i] accuracy: %.3f' % 
                  (i + 1, epochs, j + 1, float(train_size) / batch, accuracy))
            accuracy = 0.

print('Network trained in %.2f seconds' % (time.time() - start)) # 98.38 seconds

# The same network with one GPU and batches of size 300 takes ~175 seconds
# for 5 epochs, i.e. with 2 GPUs the networks trains ~1.8x faster.
# The training speed does not increase linearly in the number of GPUs because
# of the overhead due to communications between GPUs for instance.
