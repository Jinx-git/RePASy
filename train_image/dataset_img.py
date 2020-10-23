import torch
from torch.nn.functional import one_hot
from torch.utils import data
from PIL import Image
import numpy as np


class TrainDataset(data.Dataset):
    def __init__(self, file_list, channel_name, transform=None, phase="train", mic_augmentation=False, image_dim="2d"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.mic_augmentation = mic_augmentation
        self.note = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]
        self.channel_name = channel_name
        self.image_dim = image_dim

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.image_dim == "2d":
            ch_img = torch.zeros((len(self.channel_name), 128, 128))

            for i, ch_name in enumerate(self.channel_name):
                img_path = self.file_list[index][:13] + ch_name + self.file_list[index][16:]
                img1 = np.array(Image.open(img_path))
                img1 = self.transform(img1)
                if self.mic_augmentation:
                    img2 = np.array(Image.open(img_path[0:29] + "2" + img_path[30:]))
                    img2 = self.transform(img2)
                    rand = np.random.rand(2)
                    rand = rand / rand.sum()
                    # print(rand)
                    img1 = img1 * rand[0] + img2 * rand[1]
                    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
                ch_img[i] = img1

            label = (float(img_path[20:25]) - 0.5) / 0.1
            note_oh = one_hot(torch.tensor(self.note.index(img_path[17:19])), num_classes=13)
            note = self.note.index(img_path[17:19])
            return ch_img, label, note_oh.float(), note

        else:
            ch_img = torch.zeros((len(self.channel_name), 128))

            for i, ch_name in enumerate(self.channel_name):
                img_path = self.file_list[index][:13] + ch_name + self.file_list[index][16:]
                img1 = np.array(Image.open(img_path))
                # print(img1)
                img1 = np.mean(img1, axis=1).reshape(1, 128)
                # print(img1)
                img1 = self.transform(img1)
                if self.mic_augmentation:
                    img2 = np.array(Image.open(img_path[0:29] + "2" + img_path[30:]))
                    img2 = np.mean(img2, axis=1).reshape(1, 128)
                    img2 = self.transform(img2)
                    rand = np.random.rand(2)
                    rand = rand / rand.sum()
                    # print(rand)
                    img1 = img1 * rand[0] + img2 * rand[1]
                    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
                ch_img[i] = img1
                # print(ch_img)

            label = (float(img_path[20:25]) - 0.50) / 0.1
            # print(label)
            note_oh = one_hot(torch.tensor(self.note.index(img_path[17:19])), num_classes=13)
            note = self.note.index(img_path[17:19])
            return ch_img, label, note_oh.float(), note

class ValDataset(data.Dataset):
    def __init__(self, file_list, channel_name, transform=None, phase="train", image_dim="2d"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.note = ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6", "E6", "F6", "G6", "A6"]
        self.channel_name = channel_name
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
            label = (float(img_path[20:25]) - 0.5) / 0.1


            note_oh = one_hot(torch.tensor(self.note.index(img_path[17:19])), num_classes=13)
            note = self.note.index(img_path[17:19])

            return ch_img, label, note_oh.float(), note

        else:
            ch_img = torch.zeros((len(self.channel_name), 128))

            for i, ch_name in enumerate(self.channel_name):
                img_path = self.file_list[index][:13] + ch_name + self.file_list[index][16:]
                img = np.array(Image.open(img_path))
                img = np.mean(img, axis=1).reshape(1, 128)
                img = self.transform(img)
                ch_img[i] = img

            label = (float(img_path[20:25]) - 0.50) / 0.1
            # print(label)

            note_oh = one_hot(torch.tensor(self.note.index(img_path[17:19])), num_classes=13)
            note = self.note.index(img_path[17:19])

            return ch_img, label, note_oh.float(), note