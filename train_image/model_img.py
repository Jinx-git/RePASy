import torch
import torch.nn as nn

'''
class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2),
                                   nn.BatchNorm2d(8),
                                   nn.Conv2d(8, 16, 5, stride=2),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.note = nn.Sequential(nn.Dropout(p=0.5),
                                  nn.Linear(13456, 13))

        self.flow1 = nn.Sequential(nn.Linear(9216, 10),
                                   nn.ReLU())

        self.flow2 = nn.Sequential(nn.Linear(23, 1),
                                   nn.Sigmoid())

    def forward(self, x1, x2, t_note=False):
        x = self.conv1(x1)
        x_note = x.view(x.size()[0], -1)
        x = self.conv2(x)
        x_flow = x.view(x.size()[0], -1)
        flow = self.flow1(x_flow)
        note = self.note(x_note)
        if t_note:
            flow = self.flow2(torch.cat([flow, x2], dim=1))
        else:
            _, note_oh = torch.max(note.data, 1)

            flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh]], dim=1))
        return flow, note
'''


class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 32, 5, stride=2),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.Conv1d(32, 64, 3, stride=2),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   )

        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 3),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 256, 3),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU()
                                   )

        self.note = nn.Sequential(nn.Dropout(p=0.5),
                                  nn.Linear(1920, 13))

        self.flow1 = nn.Sequential(nn.Linear(6656, 10),
                                   nn.ReLU())

        self.flow2 = nn.Sequential(nn.Linear(23, 1),
                                   nn.Sigmoid())

    def forward(self, x1, x2, t_note=False):
        x = self.conv1(x1)
        x_note = x.view(x.size()[0], -1)
        x = self.conv2(x)
        # print(x.shape)
        x_flow = x.view(x.size()[0], -1)
        flow = self.flow1(x_flow)
        note = self.note(x_note)
        if t_note:
            flow = self.flow2(torch.cat([flow, x2], dim=1))
        else:
            _, note_oh = torch.max(note.data, 1)
            flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh]], dim=1))
        return flow, note

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 8, 8),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(8, 16, 8),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))

        self.note = nn.Sequential(nn.Dropout(p=0.5),
                                  nn.Linear(10816, 2048))

        self.flow1 = nn.Sequential(nn.Linear(6656, 10),
                                   nn.ReLU())

        self.flow2 = nn.Sequential(nn.Linear(23, 1),
                                   nn.Sigmoid())

        self.gap = nn.AvgPool1d(128)

    def forward(self, x1, x2, t_note=True):
        x = self.conv(x1)
        x = x.view(x.size()[0], -1)
        flow = self.flow1(x)
        note = self.note(x)
        if t_note:
            flow = self.flow2(torch.cat([flow, x2], dim=1))
        else:
            _, note_oh = torch.max(note.data, 1)
            flow = self.flow2(torch.cat([flow, torch.eye(13)[note_oh].to("cuda:0")], dim=1))
        return flow, note
'''