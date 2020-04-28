import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from ReRASy import Dataset
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 32
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0003
EPOCH = 300

# trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
"""
trainset = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=trans)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=trans)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
"""
train_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/F5/img/**/**/*.png"), transform=trans)
print(len(train_dataset))
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.Dropout = nn.Dropout(p=0.5, inplace=False)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        self.fc1 = nn.Linear(in_features=128*16*16, out_features=4096)
        self.fc2 = nn.Linear(4096, 1)
        self.fc3 = nn.Linear(4096, 13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x1 = self.fc2(x)
        x1 = self.sigmoid(x1)
        x2 = self.fc3(x)



        return [x1, x2]


device = torch.device("cuda:0")
net = Net()
#net.float()
net = net.to(device)
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

mae = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

train_loss1_value = []
train_loss2_value = []
train_mae_value = []
train_acc_value = []

for epoch in range(EPOCH):
    print("epoch{}".format(epoch+1))
    sum_loss1 = 0.0
    sum_loss2 = 0.0
    sum_mae = 0.0
    sum_corrects = 0

    net.train()
    for (inputs, labels, notes) in tqdm(trainloader):
        #inputs = torch.DoubleTensor(inputs)
        #labels = torch.DoubleTensor(labels)
        inputs, labels, notes = inputs.to(device), labels.to(device), notes.to(device)
        outputs = net(inputs)
        #print(outputs, labels)
        loss1 = criterion1(outputs[0], labels.float().view(-1, 1))
        loss2 = criterion2(outputs[1], notes)

        mae_batch = mae(outputs[0], labels.float().view(-1, 1))
        _, preds = torch.max(outputs[1], 1)

        optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer.step()

        sum_loss1 += loss1.item() * labels.size(0)
        sum_loss2 += loss2.item() * labels.size(0)
        sum_mae += mae_batch.item() * labels.size(0) * 0.1
        sum_corrects += torch.sum(preds == notes.data)

    print("[Flow] train loss:{:.7g}  train mae:{:.7g}".format(sum_loss1 / len(trainloader.dataset), sum_mae / len(trainloader.dataset)))
    train_loss1_value.append(sum_loss1 / len(trainloader.dataset))
    train_mae_value.append(sum_mae / len(trainloader.dataset))

    print("[Note] train loss:{:.7g}  train acc:{:.7g}".format(sum_loss2 / len(trainloader.dataset), sum_corrects / len(trainloader.dataset)))
    train_loss2_value.append(sum_loss2 / len(trainloader.dataset))
    train_acc_value.append(sum_corrects / len(trainloader.dataset))
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
plt.plot(range(EPOCH), train_loss1_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
plt.ylim(0, 0.005)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss1'])
plt.title('loss')
plt.savefig("loss1_image.png")
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


plt.plot(range(EPOCH), train_loss2_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
plt.ylim(0, 0.005)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss2'])
plt.title('loss')
plt.savefig("loss2_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('MAE')
plt.legend(['train acc'])
plt.title('acc')
plt.savefig("acc_image.png")
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