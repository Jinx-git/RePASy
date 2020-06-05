import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 4, 5),
                                  nn.BatchNorm2d(4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(4, 8, 5),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(8, 16, 3),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(16, 16, 3),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, stride=2)
                                  )

        self.note = nn.Sequential(nn.Linear(10816, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(2048, 13))

        self.flow1 = nn.Sequential(nn.Linear(10816, 10),
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
