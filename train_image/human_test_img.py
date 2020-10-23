import torch.onnx
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision
from PIL import Image
import glob
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt

dev = "cuda"
load_path = "../models/test/"
channel_name = ["log", "m21", "m22"]


class RecDataset(data.Dataset):
    def __init__(self, file_list, ch_name, transform=None, phase="train", image_dim="1d"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.note = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]
        self.channel_name = ch_name
        self.image_dim = image_dim

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.image_dim == "2d":
            ch_img = torch.zeros((len(self.channel_name), 128, 128))
            for i, ch_name in enumerate(self.channel_name):
                img_path = self.file_list[index][:13] + ch_name + self.file_list[index][16:]
                img = np.array(Image.open(img_path))
                img = self.transform(img)
                ch_img[i] = img
            flow = str(img_path[25])
            note = self.note.index(img_path[17:19])
            img_number = int(img_path[27:30])
            mic = str(img_path[20:24])
            # print(ch_img.shape)
            return ch_img, flow, img_number, note, mic

        if self.image_dim == "1d":
            ch_img = torch.zeros((len(self.channel_name), 128))
            for i, ch_name in enumerate(self.channel_name):
                img_path = self.file_list[index][:13] + ch_name + self.file_list[index][16:]
                img = np.array(Image.open(img_path))
                # print(img)
                img = np.mean(img, axis=1).reshape(1, 128)
                # print(img)
                img = self.transform(img)
                # print(img)
                ch_img[i] = img
            flow = str(img_path[25])
            note = self.note.index(img_path[17:19])
            img_number = int(img_path[27:30])
            mic = str(img_path[20:24])
            # print(ch_img.shape)
            return ch_img, flow, img_number, note, mic


BATCH_SIZE = 1
ACC_NUMBER = 0

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = RecDataset(file_list=glob.glob("../HumanData/mel/**/mic1/*/000.png"), ch_name=channel_name, transform=trans, image_dim="1d")
# test_dataset = RecDataset(file_list=glob.glob("../HumanData/mel/**/**/*/***.png"), ch_name=channel_name, transform=trans, image_dim="1d")
print("test data : ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 4, 8),
                                  nn.BatchNorm2d(4),
                                  nn.ReLU(),
                                  nn.Conv2d(4, 4, 2, stride=2),
                                  nn.ReLU(),
                                  # nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(4, 8, 8),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 8, 2, stride=2),
                                  nn.ReLU()
                                  # nn.MaxPool2d(2, stride=2)
                                  )

        self.note = nn.Sequential(nn.Linear(5408, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(2048, 13))

        self.flow1 = nn.Sequential(nn.Linear(5408, 10),
                                   nn.ReLU())

        self.flow2 = nn.Sequential(nn.Linear(23, 1),
                                   nn.Sigmoid())

    def forward(self, x1, x2, t_note=False):
        x = self.conv(x1)
        x = x.view(x.size()[0], -1)
        flow = self.flow1(x)
        note = self.note(x)
        if t_note:
            print("kotti")
            flow = self.flow2(torch.cat([flow, x2], dim=1))
        else:
            _, note_oh = torch.max(note.data, 1)
            # print(note_oh)
            flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh]], dim=1))
            print("else")
        return flow, note

if dev == "cpu":
    device = torch.device("cpu")
    net = Net()
    net.load_state_dict(torch.load("testmodel-nopool.m", map_location=torch.device("cpu")))
elif dev == "cuda":
    torch.nn.Module.dump_patches = True
    device = torch.device("cuda")
    net = torch.load(load_path + "pitch/model-50-epoch")
print(device)
net.eval()
criterion1 = nn.MSELoss()
weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
mae = nn.L1Loss()

for name, param in net.named_parameters():
    param.requires_grad = False

net.eval()
notes_oh = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).to(device)

# 結果保存配列の用意
note_true = []
note_predict = []
note_flow = np.zeros([13, 3])
flow_number = np.zeros([13, 3])
result = []
net.to("cpu")
for inputs, flows, img_numbers, notes, mic in test_loader:
    # print(inputs)
    # inputs = inputs.to(device)
    # print(inputs.size(), notes_oh.size())
    outputs = net(inputs, notes_oh)
    _, preds = torch.max(outputs[1], 1, False)
    note_true.append(notes.item())
    note_predict.append(preds.item())

    if preds.item() == notes.item():
        if flows[0] == "F":
            note_flow[notes.item(), 0] += outputs[0].item()
            flow_number[notes.item(), 0] += 1
        elif flows[0] == "M":
            note_flow[notes.item(), 1] += outputs[0].item()
            flow_number[notes.item(), 1] += 1
        else:
            note_flow[notes.item(), 2] += outputs[0].item()
            flow_number[notes.item(), 2] += 1
    result.append([mic[0], img_numbers.item(), flows[0], outputs[0].item()*0.1+0.50, notes.item(), preds.item()])
    print("{}, {}, {}, {:.11g} ,{} , {}".format(mic[0], img_numbers.item(), flows[0], outputs[0].item(), notes.item(), preds.item()))
    print(outputs[1])

'''
# 結果をCSVで保存
note_v = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]
note_h = [[""],
          ["C5"],
          ["D5"],
          ["E5"],
          ["F5"],
          ["G5"],
          ["A5"],
          ["B5"],
          ["C6"],
          ["D6"],
          ["E6"],
          ["F6"],
          ["G6"],
          ["A6"]]

confusion_matrix = metrics.confusion_matrix(note_true, note_predict)
cmd = metrics.ConfusionMatrixDisplay(confusion_matrix, note_v)
cmd.plot(cmap=plt.cm.Blues)
plt.savefig(load_path + "aaa.jpg")
confusion_matrix = np.vstack((note_v, confusion_matrix))
confusion_matrix = np.hstack((note_h, confusion_matrix))
np.savetxt(load_path + "confusion_matrix.csv", confusion_matrix, delimiter=",", fmt="%s")

recall = [metrics.recall_score(note_true, note_predict, average=None)]
precision = [metrics.precision_score(note_true, note_predict, average=None)]
f_measure = [metrics.f1_score(note_true, note_predict, average=None)]
score = np.concatenate([recall, precision, f_measure], axis=0).T
score = np.vstack((["Recall", "Precision", "F-Measure"], score))
score = np.hstack((note_h, score))
np.savetxt(load_path + "score.csv", score, delimiter=",", fmt="%s")

note_flow = note_flow / flow_number * 0.1 + 0.5
note_flow = np.vstack((["F", "M", "P"], note_flow))
np.savetxt(load_path + "note_flow.csv", note_flow, delimiter=",", fmt="%s")

result = np.vstack((["mic", "ImageNum", "Flow", "P_flow", "T_pitch", "P_pitch"], result))
np.savetxt(load_path + "result.csv", result, delimiter=",", fmt="%s")

acc = 0
for n in range(1, 14):
    acc += int(confusion_matrix[n, n])
acc = acc / 1248
print("ACC : ", acc)
print("F-Measure average : ", np.mean(f_measure))
'''

