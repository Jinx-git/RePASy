import numpy as np
import glob
import librosa
import librosa.display
from PIL import Image
import os

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128
data = "l22"
# data: mel, m21, m22, mfc, cqt, log, l21, l22

def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr, mono=False)

    if data == "mel":
        image1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=128, hop_length=172).T
        image2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=128, hop_length=172).T

    elif data == "m21":
        logmel256_1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=256, hop_length=172).T
        logmel256_2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=172).T
        image1 = logmel256_1[:, 0:128]
        image2 = logmel256_2[:, 0:128]

    elif data == "m22":
        logmel256_1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=256, hop_length=172).T
        logmel256_2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=172).T
        image1 = logmel256_1[:, 128:256]
        image2 = logmel256_2[:, 128:256]

    elif data == "mfc":
        image1 = librosa.feature.mfcc(y=np.array(audio[0, :]), sr=sr, n_mfcc=128, hop_length=172).T
        image2 = librosa.feature.mfcc(y=np.array(audio[1, :]), sr=sr, n_mfcc=128, hop_length=172).T

    elif data == "cqt":
        return

    elif data == "log":
        logmel1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=128, hop_length=172).T
        image1 = np.log10(logmel1 + 1e-9)
        logmel2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=128, hop_length=172).T
        image2 = np.log10(logmel2 + 1e-9)

    elif data == "l21":
        logmel256_1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=256, hop_length=172).T
        image1 = np.log10(logmel256_1 + 1e-9)[:, 0:128]
        logmel256_2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=172).T
        image2 = np.log10(logmel256_2 + 1e-9)[:, 0:128]

    elif data == "l22":
        logmel256_1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=256, hop_length=172).T
        image1 = np.log10(logmel256_1 + 1e-9)[:, 128:256]
        logmel256_2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=172).T
        image2 = np.log10(logmel256_2 + 1e-9)[:, 128:256]

    return image1, image2


def data_list(path, sr, img_path, n_mels=128):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    wav_list = glob.glob(path)
    # print(wav_list)
    for i, wavname in enumerate(wav_list):
        notename = wavname[30:32]
        flowname = wavname[32]
        # print(wavname, wavname[30:32], wavname[32])
        component1, component2 = extract_logmel(wavname, sr=sr, n_mels=128)
        # print(component.shape)
        if not os.path.exists("{}/{}".format(img_path, notename)):
            os.mkdir("{}/{}".format(img_path, notename))
        if not os.path.exists("{}/{}/mic1".format(img_path, notename)):
            os.mkdir("{}/{}/mic1".format(img_path, notename))
        if not os.path.exists("{}/{}/mic2".format(img_path, notename)):
            os.mkdir("{}/{}/mic2".format(img_path, notename))
        if not os.path.exists("{}/{}/mic1/{}".format(img_path, notename, flowname)):
            os.mkdir("{}/{}/mic1/{}".format(img_path, notename, flowname))
        if not os.path.exists("{}/{}/mic2/{}".format(img_path, notename, flowname)):
            os.mkdir("{}/{}/mic2/{}".format(img_path, notename, flowname))
        x = 0
        for n in range(0, component1.shape[0]//128):
            crop_component = component1[n*128:(n+1)*128, :]
            crop_component = crop_component + (abs(crop_component.min()))
            crop_component = crop_component / crop_component.max() * 255 // 1
            image = Image.fromarray(crop_component.T)
            image = image.convert("L")
            image.save("{}/{}/mic1/{}/{:03d}.png".format(img_path, notename, flowname, x))
            x += 1
        x = 0
        for n in range(0, component2.shape[0]//128):
            crop_component = component2[n*128:(n+1)*128, :]
            crop_component = crop_component + (abs(crop_component.min()))
            crop_component = crop_component / crop_component.max() * 255 // 1
            image = Image.fromarray(crop_component.T)
            image = image.convert("L")
            image.save("{}/{}/mic2/{}/{:03d}.png".format(img_path, notename, flowname, x))
            x += 1


# data_list("E:/RePASy/humandata/wav_human/**", SAMPLING_FREQUENCY, "E:/RePASy/humandata/" + data)
data_list("E:/RePASy/humandata/android/**", SAMPLING_FREQUENCY, "E:/RePASy/A_humandata/" + data)