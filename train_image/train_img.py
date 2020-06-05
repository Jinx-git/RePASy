# torch関連ライブラリ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
# プロット用ライブラリ
import matplotlib.pyplot as plt
# パス関連ライブラリ
import os
import glob
# 自作ライブラリ
from RePASy.train_image import dataset_img as dataset
from RePASy.train_image.model_img import Net
# 学習進捗監視用ライブラリ
from tqdm import tqdm

BATCH_SIZE = 64
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 100
LR_DOWN_EPOCH = 5
true_note = True
load_model_dir = "../models/3ch_8channel_img/flow/model-50-epoch"
save_model_dir = "../models/3ch_8channel_img/flow_2"
train = "Flow"
first = False

# datasetの読み込み
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = dataset.RecDataset(file_list=glob.glob("../ImageData/3ch_/**/img/**/**/0?[0134579].png"), transform=trans)
print("train data : ", len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = dataset.RecDataset(file_list=glob.glob("../ImageData/3ch_/**/img/**/**/0?[26].png"), transform=trans)
print("val data : ", len(val_dataset))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_dataset = dataset.RecDataset(file_list=glob.glob("../ImageData/3ch_/**/img/**/**/0?8.png"), transform=trans)
print("test data : ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

dataloaders_dict = {"train": train_loader, "val": val_loader}

# modelの読み込み
device = torch.device("cuda:0")
if first:
    net = Net()
else:
    net = torch.load(load_model_dir)
net = net.to(device)

# 学習モデルの保存場所作成
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

# Lossの設定
criterion1 = nn.MSELoss()
weight = torch.tensor([5200, 4200, 5200, 6200, 6000, 8000, 7000, 7000, 6000, 6000, 7000, 6000, 6000])
weight = torch.tensor(8000.0) / weight
criterion2 = nn.CrossEntropyLoss(weight=weight.to(device))
mae = nn.L1Loss()

# 更新するパラメータを指定
params_to_update = []
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

# Optimizerの設定
optimizer = optim.Adam(params=params_to_update, lr=LEARNING_RATE)

# 損失と評価の保存場所を作成
loss1_value = {"train": [0.], "val": [], "test": []}
loss2_value = {"train": [0.], "val": [], "test": []}
mae_value = {"train": [0.], "val": [], "test": []}
acc_value = {"train": [0.], "val": [], "test": []}

# 学習
for epoch in tqdm(range(EPOCH)):
    print("Epoch {}/{}".format(epoch+1, EPOCH))
    print('-------------')

    # 学習と検証の切り替え
    for phase in ["train", "val"]:
        if phase == "train":
            net.train()  # モデルを訓練モードに
        else:
            net.eval()  # モデルを検証モードに

        # epochの損失と評価の保存場所
        sum_loss1 = 0.0
        sum_loss2 = 0.0
        sum_mae = 0.0
        sum_corrects = 0

        # 1epoch目は検証のみ
        if (epoch == 0) and (phase == "train"):
            continue

        # iter
        for (inputs, labels, notes_oh, notes) in dataloaders_dict[phase]:

            # Optimizerのリセット
            optimizer.zero_grad()

            # 学習データをデバイスに転送
            inputs, labels = inputs.to(device), labels.to(device),
            notes_oh, notes = notes_oh.to(device), notes.to(device)

            # モデルに入力
            outputs = net(inputs, notes_oh, true_note)

            # Flow、Conv学習時の重み更新
            if train == "Flow" or train == "Conv":
                loss1 = criterion1(outputs[0], labels.float().view(-1, 1))
                mae_batch = mae(outputs[0], labels.float().view(-1, 1))
                if phase == "train":
                    loss1.backward()
                    optimizer.step()
                sum_loss1 += loss1.item() * labels.size(0)
                sum_mae += mae_batch.item() * labels.size(0) * 0.1
            # Pitch学習時の重み更新
            elif train == "Pitch":
                loss2 = criterion2(outputs[1], notes)
                _, preds = torch.max(outputs[1], 1)
                if phase == "train":
                    loss2.backward()
                    optimizer.step()
                sum_loss2 += loss2.item() * labels.size(0)
                sum_corrects += torch.sum(preds == notes.data)

        # Flow、Conv学習時のLoss表示
        if train == "Flow" or train == "Conv":
            e_loss = sum_loss1 / len(dataloaders_dict[phase].dataset)
            e_mae = sum_mae / len(dataloaders_dict[phase].dataset)
            print("[Flow : {}] loss:{:.7g}  mae:{:.7g}".format(phase, e_loss, e_mae))
            loss1_value[phase].append(e_loss)
            mae_value[phase].append(e_mae)
        # Pitch学習時のLoss表示
        elif train == "Pitch":
            e_loss = sum_loss2 / len(dataloaders_dict[phase].dataset)
            e_acc = sum_corrects.double() / len(dataloaders_dict[phase].dataset)
            print("[Pitch : {}] loss:{:.7g}  acc:{:.7g}".format(phase, e_loss, e_acc))
            loss2_value[phase].append(e_loss)
            acc_value[phase].append(e_acc)
    # 5epochごとに学習済みモデルを保存
    if not (epoch + 1) % 5:
        torch.save(net, save_model_dir + "/model-{}-epoch".format(epoch + 1))
    # 指定したepochごとにlrを減少
    # if not (epoch + 1) % LR_DOWN_EPOCH:
    #     LEARNING_RATE = LEARNING_RATE / 2.
    #     optimizer = optim.Adam(params=params_to_update, lr=LEARNING_RATE)

# 以下グラフ描画
plt.figure(figsize=(6, 6))

if train == "Flow" or train == "Conv":
    plt.plot(range(1, EPOCH), loss1_value["train"][1:])
    plt.plot(range(EPOCH), loss1_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.xlabel('EPOCH')
    plt.ylabel('MSE')
    plt.legend(['train loss1', "val loss1"])
    plt.title('loss1')
    plt.savefig(save_model_dir + "/loss1_image.png")
    plt.clf()

    plt.plot(range(1, EPOCH), mae_value["train"][1:])
    plt.plot(range(EPOCH), mae_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.xlabel('EPOCH')
    plt.ylabel('MAE')
    plt.legend(['train mae', "val mae"])
    plt.title('mae')
    plt.savefig(save_model_dir + "/mae_image.png")
    plt.clf()

elif train == "Pitch":
    plt.plot(range(1, EPOCH), loss2_value["train"][1:])
    plt.plot(range(EPOCH), loss2_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.xlabel('EPOCH')
    plt.ylabel('CROSS_ENTROPY_ERROR')
    plt.legend(['train loss2', "val loss2"])
    plt.title('loss2')
    plt.savefig(save_model_dir + "/loss2_image.png")
    plt.clf()

    plt.plot(range(1, EPOCH), acc_value["train"][1:])
    plt.plot(range(EPOCH), acc_value["val"], c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.xlabel('EPOCH')
    plt.ylabel('ACC')
    plt.legend(['train acc', "val acc"])
    plt.title('acc')
    plt.savefig(save_model_dir + "/acc_image.png")
    plt.clf()
