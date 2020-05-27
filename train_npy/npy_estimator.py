import torch
import torch.nn as nn
import torch.optim as optim
import glob
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from RePASy.train_npy import Dataset_npy as Dataset
from tqdm import tqdm

BATCH_SIZE = 64
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.00001
EPOCH = 50
LR_DOWN_EPOCH = 5
train = "Flow"
true_note = True
model = "models/npy/pitch/model-5-epoch"

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.5,), (0.5,))])
train_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/**/img/**/**/0?[0134579].npy"), transform=trans)
print("train data : ", len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/**/img/**/**/0?[26].npy"), transform=trans)
print("val data : ", len(val_dataset))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_dataset = Dataset.RecDataset(file_list=glob.glob("ImageData/**/**/img/**/**/0?8.npy"), transform=trans)
print("test data : ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

dataloaders_dict = {"train": train_loader, "val": val_loader}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 4, 5),
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
                                  )

        self.note = nn.Sequential(nn.Linear(1936, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(2048, 13))

        self.flow1 = nn.Sequential(nn.Linear(1936, 2048),
                                   nn.ReLU(),
                                   nn.Linear(2048, 10),
                                   nn.ReLU()
                                   )

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
net = torch.load(model)
net = net.to(device)

criterion1 = nn.MSELoss()
weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
mae = nn.L1Loss()
#for name, param in net.named_parameters():
    #print(name)

params_to_update = []
#update_param_names1 = ["conv.0.weight", "conv.0.bias", "conv.1.weight", "conv.1.bias", "conv.4.weight", "conv.4.bias",
                       #"flow1.0.weight", "flow1.0.bias", "flow2.0.weight", "flow2.0.bias"]
update_param_names1 = ["flow1", "flow2"]
update_param_names2 = ["note"]
update_param_names3 = ["conv"]
if train == "Flow":
    update_param_names = update_param_names1
elif train == "Pitch":
    update_param_names = update_param_names2
elif train == "Conv":
    update_param_names = update_param_names3

for name, param in net.named_parameters():
    t = 0
    for i in update_param_names:
        if i in name:
            param.requires_grad = True
            params_to_update.append(param)
            print("update:", name)
            t = 1
            break
    if t:
        continue
    param.requires_grad = False
    print("N:", name)

optimizer = optim.Adam(params=params_to_update, lr=LEARNING_RATE)

loss1_value = {"train": [0.], "val": [], "test": []}
loss2_value = {"train": [0.], "val": [], "test": []}
mae_value = {"train": [0.], "val": [], "test": []}
acc_value = {"train": [0.], "val": [], "test": []}

for epoch in range(EPOCH):
    print("Epoch {}/{}".format(epoch + 1, EPOCH))
    print('-------------')

    for phase in ["train", "val"]:
        # print(phase)
        if phase == "train":
            net.train()  # モデルを訓練モードに
        else:
            net.eval()  # モデルを検証モードに

        sum_loss1 = 0.0
        sum_loss2 = 0.0
        sum_mae = 0.0
        sum_corrects = 0

        if (epoch == 0) and (phase == "train"):
            continue

        for (inputs, labels, notes_oh, notes) in tqdm(dataloaders_dict[phase]):

            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device),
            notes_oh, notes = notes_oh.to(device), notes.to(device)

            outputs = net(inputs, notes_oh, true_note)
            # print(outputs, labels)
            if train == "Flow" or train == "Conv":
                loss1 = criterion1(outputs[0], labels.float().view(-1, 1))
                mae_batch = mae(outputs[0], labels.float().view(-1, 1))
                if phase == "train":
                    loss1.backward()
                    optimizer.step()
                sum_loss1 += loss1.item() * labels.size(0)
                sum_mae += mae_batch.item() * labels.size(0) * 0.1

            elif train == "Pitch":
                loss2 = criterion2(outputs[1], notes)
                _, preds = torch.max(outputs[1], 1)
                if phase == "train":
                    loss2.backward()
                    optimizer.step()
                sum_loss2 += loss2.item() * labels.size(0)
                sum_corrects += torch.sum(preds == notes.data)

        if train == "Flow" or train == "Conv":
            e_loss = sum_loss1 / len(dataloaders_dict[phase].dataset)
            e_mae = sum_mae / len(dataloaders_dict[phase].dataset)
            print("[Flow : {}] loss:{:.7g}  mae:{:.7g}".format(phase, e_loss, e_mae))
            loss1_value[phase].append(e_loss)
            mae_value[phase].append(e_mae)
        elif train == "Pitch":
            e_loss = sum_loss2 / len(dataloaders_dict[phase].dataset)
            e_acc = sum_corrects.double() / len(dataloaders_dict[phase].dataset)
            print("[Pitch : {}] loss:{:.7g}  acc:{:.7g}".format(phase, e_loss, e_acc))
            loss2_value[phase].append(e_loss)
            acc_value[phase].append(e_acc)


    if not (epoch + 1) % 5:
        torch.save(net, "model-{}-epoch".format(epoch + 1))

    # if not (epoch + 1) % LR_DOWN_EPOCH:
        # LEARNING_RATE = LEARNING_RATE / 2.
        # optimizer = optim.Adam(params=params_to_update, lr=LEARNING_RATE)
"""
    sum_loss = 0.0
    sum_correct = 0

    net.eval()
    for (inputs, labels) in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        sum_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        sum_correct += (predicted == labels).sum().item()
    print("test loss:{}, accuracy:{}".format(sum_loss / len(testloader.dataset),
                                             float(sum_correct / len(testloader.dataset))))
    test_loss_value.append(sum_loss / len(testloader.dataset))
    test_acc_value.append(float(sum_correct / len(testloader.dataset)))
"""

plt.figure(figsize=(6, 6))      #グラフ描画用

#以下グラフ描画
if train == "Flow" or train == "Conv":
    plt.plot(range(1, EPOCH), loss1_value["train"][1:])
    plt.plot(range(EPOCH), loss1_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    #plt.ylim(0, np.max(train_loss_value))
    #plt.ylim(0, 0.005)
    plt.xlabel('EPOCH')
    plt.ylabel('MSE')
    plt.legend(['train loss1', "val loss1"])
    plt.title('loss1')
    plt.savefig("loss1_image.png")
    plt.clf()

    plt.plot(range(1, EPOCH), mae_value["train"][1:])
    plt.plot(range(EPOCH), mae_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    #plt.ylim(0, np.max(train_loss_value))
    #plt.ylim(0, 0.001)
    plt.xlabel('EPOCH')
    plt.ylabel('MAE')
    plt.legend(['train mae', "val mae"])
    plt.title('mae')
    plt.savefig("mae_image.png")
    plt.clf()

elif train == "Pitch":
    plt.plot(range(1, EPOCH), loss2_value["train"][1:])
    plt.plot(range(EPOCH), loss2_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    #plt.ylim(0, np.max(train_loss_value))
    #plt.ylim(0, 3)
    plt.xlabel('EPOCH')
    plt.ylabel('CROSS_ENTROPY_ERROR')
    plt.legend(['train loss2', "val loss2"])
    plt.title('loss2')
    plt.savefig("loss2_image.png")
    plt.clf()

    plt.plot(range(1, EPOCH), acc_value["train"][1:])
    plt.plot(range(EPOCH), acc_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    #plt.ylim(0, np.max(acc_value["val"]))
    #plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACC')
    plt.legend(['train acc', "val acc"])
    plt.title('acc')
    plt.savefig("acc_image.png")
    plt.clf()
