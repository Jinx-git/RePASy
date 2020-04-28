from torch.utils import data
import torchvision
from PIL import Image
import glob
import numpy as np
from torch.nn.functional import one_hot
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
        img = self.transform(img)
        img = img / 255.0
        label = (float(img_path[22:27]) - 0.5) / 0.1
        note = self.note.index(img_path[15:17])
        #print(note)
        #print(label)
        #label = (float(img_path[27:31]))
        label = np.array(label)
        return img, label, note
