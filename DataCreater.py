import numpy as np
import glob
import librosa
import librosa.display
import sys
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128
DATA_PATH = "Data/rec1"


def extract_logmel(wav, sr, n_mels=128): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    logmel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=172).T
    return logmel


def generate_image(path, sr, img_path, n_mels=128):
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))

    for i, wavname in enumerate(wav_list):
        component = extract_logmel(wavname, sr=sr, n_mels=128)
        for n in range(0, component.shape[0]//128):
            crop_component = component[n*128:(n+1)*128, :]
            crop_component = crop_component / crop_component.max() * 255 // 1
            image = Image.fromarray(crop_component.T)
            image = image.convert("L")
            image.save("{}/{:03d}.png".format(img_path, 20*i+n))


basename = os.path.basename(DATA_PATH)
print("BaseName : " + basename)
img_path = os.path.join("ImageData/np", basename)
if not os.path.exists(img_path):
    os.mkdir(img_path)

note_list = glob.glob(DATA_PATH + "/*")
for note_path in tqdm(note_list):
    note_name = os.path.basename(note_path)
    img_note_path = os.path.join(img_path, note_name)
    if not os.path.exists(img_note_path):
        os.mkdir(img_note_path)
        os.mkdir(img_note_path + "/img")
        os.mkdir(img_note_path + "/Visible_Image")
    img_note_data_path = os.path.join(img_note_path, "img")
    img_note_visible_path = os.path.join(img_note_path, "Visible_Image")

    flow_list = glob.glob(note_path + "/*")
    for flow_path in tqdm(flow_list):
        flow_name = os.path.basename(flow_path)
        img_flow_path = os.path.join(img_note_data_path, flow_name)
        if not os.path.exists(img_flow_path):
            os.mkdir(img_flow_path)

        mic_list = glob.glob(flow_path + "/*")
        for i, mic_path in enumerate(mic_list):
            mic_name = os.path.basename(mic_path)
            img_mic_path = os.path.join(img_flow_path, mic_name)
            if not os.path.exists(img_mic_path):
                os.mkdir(img_mic_path)
            generate_image(mic_path + "/*", SAMPLING_FREQUENCY, img_mic_path, n_mels=NUMBER_MEL)
