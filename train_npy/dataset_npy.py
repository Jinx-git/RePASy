import torch
from torch.nn.functional import one_hot
from torch.utils import data
import numpy as np


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
        img = np.load(img_path)

        img = self.transform(img)
        label = (float(img_path[25:30]) - 0.5) / 0.1

        note_oh = one_hot(torch.tensor(self.note.index(img_path[18:20])), num_classes=13)
        note = self.note.index(img_path[18:20])

        return img, label, note_oh.float(), note
