import torch
import torch.onnx
import torch.nn as nn
import torchvision
import numpy as np
import sounddevice as sd
import librosa
from PIL import Image
import matplotlib.pyplot as plt


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

    def forward(self, x1):
        x = self.conv(x1)
        x = x.view(x.size()[0], -1)
        flow = self.flow1(x)
        note = self.note(x)
        _, note_oh = torch.max(note.data, 1)
        flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh]], dim=1))
        return flow, note


def extract_logmel():
    audio = sd.rec(int(0.5 * 44100))
    sd.wait()
    logmel = librosa.feature.melspectrogram(y=audio.reshape(-1), sr=44100, n_mels=128, hop_length=173).T[0:128][:]
    logmel = logmel / logmel.max() * 255 // 1
    image = Image.fromarray(logmel.T)
    image = image.convert("L")
    image.save("test.png")
    return image

print(sd.query_devices())
sd.default.samplerate = 44100
sd.default.channels = 1
dev = [1, 3]

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

net = Net()
net.load_state_dict(torch.load("testmodel.m", map_location=torch.device("cpu")))
net.eval()

note = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]
arr = np.zeros(50)
label = []
for i in range(50):
    label.append(" ")

x = np.arange(0, 50, 1)
plt.figure(figsize=(10,6))
plt.ylim([0.5, 0.6])
lines, = plt.plot(x, arr) #グラフオブジェクトを受け取る
i = 49
while 1:
    spec = np.array(extract_logmel())
    spec = trans(spec)
    spec = spec / 255.0
    outputs = net(spec.reshape(-1, 1, 128, 128))
    _, preds = torch.max(outputs[1], 1)
    pred = note[preds.item()]
    # print(outputs[0])
    if i < 39:
        label.append(pred)
        label = label[1:]
        i = 49
    elif label[i] == pred:
        i -= 1
        label.append(" ")
        label = label[1:]
    else:
        label.append(pred)
        label = label[1:]
        i = 49

    arr = np.append(arr, outputs[0].item()*0.1+0.5)
    arr = arr[1:]
    lines.set_data(x, arr)  # データ更新
    plt.xticks(x, label)
    plt.pause(0.0001)
    print("\r{}:{}".format(pred, outputs[0].item() * 0.1 + 0.5), end="")
