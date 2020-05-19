import numpy as np
import glob
import librosa
import librosa.display
from PIL import Image
import os

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128


def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    logmel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=173).T
    return logmel


def data_list(path, sr, img_path, n_mels=128):
    os.mkdir(img_path)
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))
    count= 0
    print(wav_list)
    x = 0
    for i, wavname in enumerate(wav_list):
        component = extract_logmel(wavname, sr=sr, n_mels=128)
        for n in range(0, component.shape[0]//128):
            crop_component = component[n*128:(n+1)*128, :]
            print(crop_component.shape)
            crop_component = crop_component / crop_component.max() * 255 // 1
            image = Image.fromarray(crop_component.T)
            image = image.convert("L")
            image.save("{}/{:03d}.png".format(img_path, x))
            x += 1

data_list("HumanData/WAV/c6m*", SAMPLING_FREQUENCY, "HumanData/img/C6/m")




