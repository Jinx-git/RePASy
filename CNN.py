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
LEARNING_RATE = 0.001
EPOCH = 30
test = True
train = 2
true_note = True

# trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
"""
trainset = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=trans)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=trans)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
"""
train_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/**/img/**/**/0??.png"), transform=trans)
print(len(train_dataset))
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 15), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(16, 32, 10), nn.ReLU(), nn.MaxPool2d(2, stride=2))

        self.note = nn.Sequential(nn.Linear(32*24*24, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 13))

        self.flow1 = nn.Sequential(nn.Linear(32*24*24, 10), nn.ReLU())
        self.flow2 = nn.Sequential(nn.Linear(23, 1), nn.Sigmoid())

    def forward(self, x1, x2, t_note):
        x = self.conv(x1)
        x = x.view(x.size()[0], -1)
        flow = self.flow1(x)
        note = self.note(x)
        if t_note:
            flow = self.flow2(torch.cat([flow, x2], dim=1))
        else:
            _, note_oh = torch.max(note.data, 1)
            # print(note_oh)
            flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh].to("cuda:0")], dim=1))

        return flow, note


device = torch.device("cuda:0")
# net = Net()
net = torch.load("models/model-30-epoch")

weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
net = net.to(device)
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
# criterion2 = nn.CrossEntropyLoss()
mae = nn.L1Loss()
params_to_update = []
update_param_names1 = ["conv.0.weight", "conv.0.bias", "conv.1.weight", "conv.1.bias", "conv.4.weight", "conv.4.bias",
                       "flow1.0.weight", "flow1.0.bias", "flow2.0.weight", "flow2.0.bias"]
update_param_names2 = ["note.0.weight", "note.0.bias", "note.3.weight", "note.3.bias"]
if train == 1:
    update_param_names = update_param_names1
elif train == 2:
    update_param_names = update_param_names2

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print("update:", name)
    else:
        param.requires_grad = False
        print("N:", name)

optimizer = optim.Adam(params=params_to_update, lr=LEARNING_RATE)

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
    for (inputs, labels, notes_oh, notes) in tqdm(trainloader):
        inputs, labels, notes_oh, notes = inputs.to(device), labels.to(device), notes_oh.to(device), notes.to(device)
        outputs = net(inputs, notes_oh, true_note)
        # print(outputs, labels)
        if train == 1:
            loss1 = criterion1(outputs[0], labels.float().view(-1, 1))
            mae_batch = mae(outputs[0], labels.float().view(-1, 1))
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer.step()
            sum_loss1 += loss1.item() * labels.size(0)
            sum_mae += mae_batch.item() * labels.size(0) * 0.1
        elif train == 2:
            loss2 = criterion2(outputs[1], notes)
            _, preds = torch.max(outputs[1], 1)
            optimizer.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer.step()
            sum_loss2 += loss2.item() * labels.size(0)
            sum_corrects += torch.sum(preds == notes.data)
    if not (epoch+1) % 5:
        torch.save(net, "model-{}-epoch".format(epoch+1))
    if train == 1:
        print("[Flow] train loss:{:.7g}  train mae:{:.7g}".format(sum_loss1 / len(trainloader.dataset),
                                                                  sum_mae / len(trainloader.dataset)))
        train_loss1_value.append(sum_loss1 / len(trainloader.dataset))
        train_mae_value.append(sum_mae / len(trainloader.dataset))
    elif train == 2:
        print("[Note] train loss:{:.7g}  train acc:{:.7g}".format(sum_loss2 / len(trainloader.dataset), sum_corrects.double() / len(trainloader.dataset)))
        train_loss2_value.append(sum_loss2 / len(trainloader.dataset))
        train_acc_value.append(sum_corrects.double() / len(trainloader.dataset))
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
""""""
if train == 1:
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
    plt.ylim(0, 0.001)
    plt.xlabel('EPOCH')
    plt.ylabel('MAE')
    plt.legend(['train mae'])
    plt.title('mae')
    plt.savefig("mae_image.png")
    plt.clf()

elif train == 2:
    plt.plot(range(EPOCH), train_loss2_value)
#plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
    plt.xlim(0, EPOCH)
#plt.ylim(0, np.max(train_loss_value))
    plt.ylim(0, 3)
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