import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# Hyper-parameters
sequence_length = 56
input_size = 14
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# initial device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# laod dataset
dataroot_MNIST = '/data/dataset/MNIST/'
trainset_MNIST = torchvision.datasets.MNIST(root=dataroot_MNIST,
                                        train=True, download=False,
                                        transform=transforms.ToTensor())
testset_MNIST = torchvision.datasets.MNIST(root=dataroot_MNIST,
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset_MNIST,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
test_loader =  torch.utils.data.DataLoader(trainset_MNIST,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

# build my lstm model

class LstmSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LstmSimple, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gatei = nn.Linear(input_size + hidden_size, hidden_size)
        self.gateo = nn.Linear(input_size + hidden_size, hidden_size)
        self.gatef = nn.Linear(input_size + hidden_size, hidden_size)
        self.cellg = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x_all): # x_all:batch_size * seq_length * input_size
        # set initial states
        self.h0 = torch.zeros(x_all.size(0), self.hidden_size).to(device)
        self.c0 = torch.zeros(x_all.size(0), self.hidden_size).to(device)
        for i in range(x_all.size(1)):
            self.step(x_all[:, i, :])
        out = self.fc(self.h0)
        # out = self.softmax(out)
        return out

    def step(self, x): # this give one time step data, for 100*28*28
                       # we can have batch_size*input_size tensor at a time
        # forward lstm, many to one
        combined = torch.cat((x, self.h0), 1)
        f_gate = self.sigmoid(self.gatef(combined))
        o_gate = self.sigmoid(self.gateo(combined))
        i_gate = self.sigmoid(self.gatei(combined))
        cellg = self.tanh(self.cellg(combined))
        self.c0 = torch.add(torch.mul(self.c0,  f_gate), torch.mul(cellg, i_gate))
        self.h0 = torch.mul(self.c0, o_gate)

model_mine = LstmSimple(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer_mine = torch.optim.SGD(model_mine.parameters(), lr=learning_rate)

# train my model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model_mine(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer_mine.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# test my model
with torch.no_grad():
    correct = 0
    total = 0

    images = images.reshape(-1, sequence_length, input_size).to(device)
    labels = labels.to(device)
    outputs = model_mine(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
