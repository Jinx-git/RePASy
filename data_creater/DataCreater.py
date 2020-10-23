import numpy as np
import glob
import librosa
import librosa.display
from PIL import Image
import os
from tqdm import tqdm

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128
DATA_PATH = "E:/RePASy/traindata"
data = "l22"
# data: mel, m21, m22, mfc, cqt, log, l21, l22

def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    if data == "mel":
        image = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=172).T
    elif data == "m21":
        logmel256 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=172).T
        image = logmel256[:, 0:128]
    elif data == "m22":
        logmel256 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=172).T
        image = logmel256[:, 128:256]
    elif data == "mfc":
        image = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128, hop_length=172).T
    elif data == "cqt":
        return
    elif data == "log":
        image = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=172).T
        image = np.log10(image + 1e-9)
    elif data == "l21":
        logmel256 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=172).T
        image = np.log10(logmel256 + 1e-9)[:, 0:128]
    elif data == "l22":
        logmel256 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=172).T
        image = np.log10(logmel256 + 1e-9)[:, 128:256]

    return image


def generate_image(path, sr, img_path, n_mels=128):
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))

    for i, wavname in enumerate(wav_list):
        component = extract_logmel(wavname, sr=sr, n_mels=128)
        for n in range(0, component.shape[0]//128):
            crop_component = component[n*128:(n+1)*128, :]
            # print(crop_component.min(), crop_component.max())
            crop_component = crop_component + (abs(crop_component.min()))
            # print(crop_component.min(), crop_component.max())
            crop_component = crop_component/ crop_component.max() * 255 // 1
            # print(crop_component.min(), crop_component.max())
            image = Image.fromarray(crop_component.T)
            image = image.convert("L")
            image.save("{}/{:03d}.png".format(img_path, 20*i+n))


img_path = os.path.join(DATA_PATH, data)
if not os.path.exists(img_path):
    os.mkdir(img_path)
wav_path = os.path.join(DATA_PATH, "wav_train")
note_list = glob.glob(wav_path + "/*")
for note_path in tqdm(note_list):
    note_name = os.path.basename(note_path)
    img_note_path = os.path.join(img_path, note_name)
    if not os.path.exists(img_note_path):
        os.mkdir(img_note_path)

    flow_list = glob.glob(note_path + "/*")
    for flow_path in tqdm(flow_list):
        flow_name = os.path.basename(flow_path)
        img_flow_path = os.path.join(img_note_path, flow_name)
        if not os.path.exists(img_flow_path):
            os.mkdir(img_flow_path)
#
        mic_list = glob.glob(flow_path + "/*")
        for i, mic_path in enumerate(mic_list):
            mic_name = os.path.basename(mic_path)
            img_mic_path = os.path.join(img_flow_path, mic_name)
            if not os.path.exists(img_mic_path):
                os.mkdir(img_mic_path)
            generate_image(mic_path + "/*", SAMPLING_FREQUENCY, img_mic_path, n_mels=NUMBER_MEL)
