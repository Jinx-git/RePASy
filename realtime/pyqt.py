# -*- coding:utf-8 -*-
#putorch関係のライブラリ
import torch
import torch.onnx
import torch.nn as nn
import torchvision

from PIL import Image

#プロット関係のライブラリ
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

#音声関係のライブラリ
import pyaudio
import librosa


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


class PlotWindow:
    def __init__(self, net):
        #モデル初期設定
        self.net = net
        self.net.load_state_dict(torch.load("testmodel.m", map_location=torch.device("cpu")))
        self.net.eval()
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        #プロット初期設定
        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle(u"リアルタイムプロット")
        self.plt=self.win.addPlot() #プロットのビジュアル関係
        self.plt.setYRange(0.5, 0.6)    #y軸の上限、下限の設定
        self.curve=self.plt.plot()  #プロットデータを入れる場所

        #マイクインプット設定
        self.CHUNK=1500             #1度に読み取る音声のデータ幅
        self.RATE=44100             #サンプリング周波数
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)

        #アップデート時間設定
        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)    #10msごとにupdateを呼び出し

        #音声データの格納場所(プロットデータ)
        self.data = np.zeros(200)
        self.spec = np.zeros((128, 128))
        self.sound = np.zeros(22050)

    def extract_logmel(self):
        # print(audio.shape)
        logmel = librosa.feature.melspectrogram(y=self.sound, sr=44100, n_mels=128, hop_length=173).T
        return logmel

    def spec2tensor(self):
        spec_input = self.spec / self.spec.max() * 255.0 // 1
        image = Image.fromarray(spec_input)
        image = image.convert("L")
        image.save("test.png")
        spec_tensor = self.trans(np.array(image).T)
        spec_tensor = spec_tensor / 255.0
        return spec_tensor

    def predict(self, input_tensor):
        outputs = self.net(input_tensor.reshape(-1, 1, 128, 128))
        return outputs[0].item()

    def update(self):
        self.AudioInput()
        input_tensor = self.spec2tensor()
        # print(self.spec.shape)
        output = self.predict(input_tensor)
        self.data = np.append(self.data, output * 0.1 + 0.5)[1:]
        self.curve.setData(self.data)   #プロットデータを格納

    def AudioInput(self):
        ret=self.stream.read(self.CHUNK)
        ret = np.frombuffer(ret, dtype="int16")/32768.0
        self.sound = np.append(self.sound, ret)[1500:]
        self.spec = self.extract_logmel()

if __name__=="__main__":
    net = Net()
    plotwin=PlotWindow(net)
    plotwin.update()
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()