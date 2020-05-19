import torch
import torch.onnx
import torch.nn as nn
import torchvision
import numpy as np
import librosa
import pyaudio
import matplotlib.pyplot as plt
import sounddevice as sd
from PIL import Image


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


def extract_logmel(audio):
    # print(audio.shape)
    logmel = librosa.feature.melspectrogram(y=np.array(audio), sr=44100, n_mels=128, hop_length=173).T
    return logmel


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024,
                input_device_index=1)
for x in range(0, p.get_device_count()):
        print(p.get_device_info_by_index(x))
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
spec = np.zeros((128, 128), dtype=np.float)
dev = [1, 3]
sd.default.channels = 1
sd.default.device = dev
sd.default.samplerate = 44100
xxx = 5
while True:
    record = sd.rec(int(44100 * 0.1))
    sd.wait()
    # audio = stream.read(int(44100 * 0.1))
    # audio = np.frombuffer(audio, dtype=np.float)
    spectrum = np.array([extract_logmel(record.reshape(-1))]).reshape(-1, 128)[3:23, :]
    # print(spectrum.shape)
    spec = np.append(spec, spectrum, axis=0)
    spec = spec[20:, :]
    spec_input = spec / spec.max() * 255.0 // 1
    # image = Image.fromarray(spectrum / spectrum.max() * 255 // 1)
    # image = image.convert("L")
    image = Image.fromarray(spec_input)
    image = image.convert("L")
    image.save("test.png")
    spec_tensor = trans(np.array(image).T)
    spec_tensor = spec_tensor / 255.0
    outputs = net(spec_tensor.reshape(-1, 1, 128, 128))
    _, preds = torch.max(outputs[1], 1)
    pred = note[preds.item()]
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
    if xxx == 0:
        lines.set_data(x, arr)  # データ更新
        plt.xticks(x, label)
        plt.pause(0.0001)
        xxx = 5
    xxx -= 1