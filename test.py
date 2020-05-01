import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from ReRASy import Dataset
from tqdm import tqdm
import numpy as np
from torchviz import make_dot

BATCH_SIZE = 1
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/A6/img/**/**/0?8.png"), transform=trans)
print("test data : ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 4, 8),
                                  nn.BatchNorm1d(4),
                                  nn.ReLU(),
                                  nn.Conv1d(4, 8, 4),
                                  # nn.BatchNorm1d(8),
                                  nn.ReLU(),
                                  )

        self.note = nn.Sequential(nn.Linear(944, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(2048, 13))

        self.flow1 = nn.Sequential(nn.Linear(944, 10),
                                   nn.ReLU())

        self.flow2 = nn.Sequential(nn.Linear(23, 1),
                                   nn.Sigmoid())

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
net = torch.load("models/compM/model-50-epoch")
net = net.to(device)

criterion1 = nn.MSELoss()
weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
mae = nn.L1Loss()

for name, param in net.named_parameters():
    param.requires_grad = False

#print(net)
net.eval()
sum_loss1 = 0.0
sum_loss2 = 0.0
sum_mae = 0.0
sum_corrects = 0
print("Tflow, Pflow, MAE, Tpitch, Ppitch")
for (inputs, labels, notes_oh, notes) in test_loader:
    inputs, labels = inputs.to(device), labels.to(device),
    notes_oh, notes = notes_oh.to(device), notes.to(device)

    outputs = net(inputs, notes_oh, False)

    loss1 = criterion1(outputs[0], labels.float().view(-1, 1))
    loss2 = criterion2(outputs[1], notes)
    _, preds = torch.max(outputs[1], 1)
    mae_batch = mae(outputs[0], labels.float().view(-1, 1))
    sum_loss1 += loss1.item() * labels.size(0)
    sum_mae += mae_batch.item() * labels.size(0) * 0.1
    sum_loss2 += loss2.item() * labels.size(0)
    sum_corrects += torch.sum(preds == notes.data)
    print("{:.7g} , {:.7g} , {:.7g} ,{} , {}".format(labels.item()*0.1+0.5, outputs[0].item()*0.1+0.5, mae_batch.item() * labels.size(0) * 0.1, notes.item(), preds.item()))


e_loss = sum_loss1 / len(test_loader.dataset)
e_mae = sum_mae / len(test_loader.dataset)
print("[Flow : {}] loss:{:.7g}  mae:{:.7g}".format("Test", e_loss, e_mae))
e_loss = sum_loss2 / len(test_loader.dataset)
e_acc = sum_corrects.double() / len(test_loader.dataset)
print("[Pitch : {}] loss:{:.7g}  acc:{:.7g}".format("Test", e_loss, e_acc))





