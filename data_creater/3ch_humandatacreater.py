import numpy as np
import glob
import librosa
import librosa.display
from PIL import Image
import os

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128
data = "img"

def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr, mono=False)
    logmel1 = librosa.feature.melspectrogram(y=np.array(audio[0, :]), sr=sr, n_mels=n_mels, hop_length=173).T
    logmel2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=n_mels, hop_length=173).T
    logmel256_1 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=173).T
    logmel256_2 = librosa.feature.melspectrogram(y=np.array(audio[1, :]), sr=sr, n_mels=256, hop_length=173).T
    logmel1_1 = logmel1.reshape([1, -1, 128])
    logmel1_2 = logmel256_1[:, 0:128].reshape([1, -1, 128])
    logmel1_3 = logmel256_1[:, 128:256].reshape([1, -1, 128])
    logmel2_1 = logmel2.reshape([1, -1, 128])
    logmel2_2 = logmel256_2[:, 0:128].reshape([1, -1, 128])
    logmel2_3 = logmel256_2[:, 128:256].reshape([1, -1, 128])
    logmel_rgb1 = np.concatenate((logmel1_1, logmel1_2, logmel1_3), axis=0)
    logmel_rgb2 = np.concatenate((logmel2_1, logmel2_2, logmel2_3), axis=0)
    if data == "npy":
        logmel1 = librosa.power_to_db(logmel1, ref=np.max)
        logmel2 = librosa.power_to_db(logmel2, ref=np.max)
    return logmel_rgb1, logmel_rgb2


def data_list(path, sr, img_path, n_mels=128):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    wav_list = glob.glob(path)
    # print(wav_list)
    for i, wavname in enumerate(wav_list):
        notename = wavname[17:19]
        flowname = wavname[19]
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
        for n in range(0, component1.shape[1]//128):
            crop_component = component1[:, n*128:(n+1)*128, :]
            if data == "npy":
                np.save("{}/{}/mic1/{}/{:03d}".format(img_path, notename, flowname, x), crop_component)
            else:
                max_rgb = crop_component.max(axis=(1, 2)).reshape([-1, 1, 1])
                crop_component = crop_component / max_rgb * 255 // 1
                image = Image.fromarray(np.uint8(crop_component.T))
                image = image.convert("RGB")
                image.save("{}/{}/mic1/{}/{:03d}.png".format(img_path, notename, flowname, x))
            x += 1
        x = 0
        for n in range(0, component2.shape[1]//128):
            crop_component = component2[:, n*128:(n+1)*128, :]
            if data == "npy":
                np.save("{}/{}/mic2/{}/{:03d}".format(img_path, notename, flowname, x), crop_component)
            else:
                max_rgb = crop_component.max(axis=(1, 2)).reshape([-1, 1, 1])
                crop_component = crop_component / max_rgb * 255 // 1
                image = Image.fromarray(np.uint8(crop_component.T))
                image = image.convert("RGB")
                image.save("{}/{}/mic2/{}/{:03d}.png".format(img_path, notename, flowname, x))
            x += 1

data_list("../HumanData/WAV/**", SAMPLING_FREQUENCY, "../HumanData/img")
