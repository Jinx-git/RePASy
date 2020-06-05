import numpy as np
import glob
import librosa
import librosa.display
from PIL import Image
import os
from tqdm import tqdm

SAMPLING_FREQUENCY = 44100
DATA_PATH = "E:/RePASy/traindata/wav_train/rec1"
data = "img"


def extract_logmel(wav, sr): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    logmel1 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=172).T
    logmel256 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=172).T
    # print(logmel256.shape)
    logmel1 = logmel1.reshape([1, -1, 128])
    logmel2 = logmel256[:, 0:128].reshape([1, -1, 128])
    logmel3 = logmel256[:, 128:256].reshape([1, -1, 128])
    # print(logmel2.shape, logmel3.shape)
    logmel_rgb = np.concatenate((logmel1, logmel2, logmel3), axis=0)
    # print(logmel_rgb.shape)
    if data == "npy":
        logmel1 = librosa.power_to_db(logmel1, ref=np.max)
        logmel2 = librosa.power_to_db(logmel2, ref=np.max)
    return logmel_rgb


def generate_image(path, sr, img_path, n_mels=128):
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))

    for i, wavname in enumerate(wav_list):
        component = extract_logmel(wavname, sr=sr)
        for n in range(0, component.shape[1]//128):
            # crop_component = component[n*128:(n+1)*128, :, :]
            crop_component = component[:, n*128:(n+1)*128, :]
            # print(crop_component.shape)
            # print(crop_component.shape)
            if data == "npy":
                np.save("{}/{:03d}".format(img_path, 20*i+n), crop_component)
            else:
                max_rgb = crop_component.max(axis=(1, 2)).reshape([-1, 1, 1])
                array = crop_component / max_rgb * 255 // 1
                arru1 = np.uint8(np.asarray(array))
                # print(arru1.shape)
                image = Image.fromarray(arru1.T)
                image = image.convert("RGB")
                image.save("{}/{:03d}.png".format(img_path, 20*i+n))


basename = os.path.basename(DATA_PATH)
print("BaseName : " + basename)
img_path = os.path.join("../ImageData_rgb", basename)
if not os.path.exists(img_path):
    os.mkdir(img_path)

note_list = glob.glob(DATA_PATH + "/*")
for note_path in tqdm(note_list):
    note_name = os.path.basename(note_path)
    img_note_path = os.path.join(img_path, note_name)
    if not os.path.exists(img_note_path):
        os.mkdir(img_note_path)
        os.mkdir(img_note_path + "/img")
    img_note_data_path = os.path.join(img_note_path, "img")

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
            generate_image(mic_path + "/*", SAMPLING_FREQUENCY, img_mic_path)
