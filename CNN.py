import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from ReRASy import Dataset
import numpy as np

BATCH_SIZE = 32
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0003
EPOCH = 500

# trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
"""
trainset = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=trans)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=trans)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
"""
train_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/**/Data/**/*.png"), transform=trans)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 15)
        self.conv2 = nn.Conv2d(16, 32, 10)

        self.fc1 = nn.Linear(32 * 24 * 24, 1)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


device = torch.device("cuda:0")
net = Net()
#net.float()
net = net.to(device)
criterion = nn.MSELoss()
mae = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

train_loss_value = []
train_mae_value = []
test_loss_value = []
test_acc_value = []

for epoch in range(EPOCH):
    print("epoch{}".format(epoch+1))
    sum_loss = 0.0
    sum_mae = 0.0
    t = 0

    net.train()
    for (inputs, labels) in trainloader:
        #inputs = torch.DoubleTensor(inputs)
        #labels = torch.DoubleTensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs, labels)
        loss = criterion(outputs, labels.float().view(-1, 1))
        mae_batch = mae(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * labels.size(0)
        sum_mae += mae_batch.item() * labels.size(0)
        t += labels.size(0)
    print("train loss:{:.7g}  train mae:{:.7g}".format(sum_loss / len(trainloader.dataset), sum_mae / len(trainloader.dataset)))
    train_loss_value.append(sum_loss / len(trainloader.dataset))
    train_mae_value.append(sum_mae / len(trainloader.dataset))
"""
    sum_loss = 0.0
    sum_correct = 0

    net.eval()
    for (inputs, labels) in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        sum_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        sum_correct += (predicted == labels).sum().item()
    print("test loss:{}, accuracy:{}".format(sum_loss / len(testloader.dataset),
                                             float(sum_correct / len(testloader.dataset))))
    test_loss_value.append(sum_loss / len(testloader.dataset))
    test_acc_value.append(float(sum_correct / len(testloader.dataset)))
"""
plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
plt.ylim(0, 0.005)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_mae_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
plt.ylim(0, 0.1)
plt.xlabel('EPOCH')
plt.ylabel('MAE')
plt.legend(['train mae'])
plt.title('mae')
plt.savefig("mae_image.png")
plt.clf()


"""
plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")

"""