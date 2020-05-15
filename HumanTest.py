import torch.onnx
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision
from PIL import Image
import glob
import numpy as np
import torch


class RecDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.note = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = np.array(Image.open(img_path))
        # flow = int(img_path[13:14])
        flow = 1
        img = self.transform(img)
        img = img / 255.0
        img_number = int(img_path[15:18])
        return img, flow, img_number


BATCH_SIZE = 1
ACC_NUMBER = 0

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = RecDataset(file_list=glob.glob("HumanData/G5te/**"), transform=trans)
print("test data : ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 4, 8),
                                  nn.BatchNorm2d(4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(4, 8, 8),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2)
                                  )

        self.note = nn.Sequential(nn.Linear(5408, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(2048, 13))

        self.flow1 = nn.Sequential(nn.Linear(5408, 10),
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
net = torch.load("models/conv2d_FP/Flow2cmp/model-50-epoch")
net = net.to(device)
torch.save(net.state_dict(), "testmodel.m")
criterion1 = nn.MSELoss()
weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
mae = nn.L1Loss()

for name, param in net.named_parameters():
    param.requires_grad = False

#print(net)
net.eval()
print("ImageNum, FlowNum, Pflow, Tpitch, Ppitch")
notes_oh = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device)
for inputs, flows, img_numbers in test_loader:
    print(inputs)
    inputs = inputs.to(device)
    outputs = net(inputs, notes_oh, False)
    _, preds = torch.max(outputs[1], 1)
    print("{}, {}, {:.7g} ,{} , {}".format(img_numbers.item(),flows.item(), outputs[0].item()*0.1+0.5, ACC_NUMBER, preds.item()))