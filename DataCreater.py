import numpy as np
import glob
import librosa
import librosa.display
import sys
from PIL import Image
import os
import matplotlib.pyplot as plt

SAMPLING_FREQUENCY = 44100
NUMBER_MEL = 128
DATA_PATH = "RecorderData/original"


def extract_logmel(wav, sr, n_mels=64): #Output -> (timeframe, logmel_dim)
    audio, _ = librosa.load(wav, sr=sr)
    #logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)).T
    logmel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=173).T
    #print(logmel.shape)
    return logmel


def data_list(path, sr, note, flow, n_mels=64):
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))
    count= 0
    for i, wavname in enumerate(wav_list):
        component = extract_logmel(wavname, sr=sr, n_mels=n_mels)

        component = component / component.max() * 255 // 1
        image = Image.fromarray(component.T)
        image = image.convert("L")
        image.save("ImageData/" + basename + "/" + note_name + "/Data/" + flow_name + "/" + str(i) + ".png")

        data = np.concatenate([data, component], axis=0)
        count += 1
        sys.stdout.write("\r%s" % note + " " + flow + ": 現在"+str(np.around((count/len(wav_list))*100 , 2))+"%完了")
        sys.stdout.flush()
    return data.T, size


basename = os.path.basename(DATA_PATH)
print("BaseName : " + basename)
if not os.path.exists("ImageData/" + basename):
    os.mkdir("ImageData/" + basename)
note_list = glob.glob(DATA_PATH + "/*")
for note_path in note_list:
    note_name = os.path.basename(note_path)
    if not os.path.exists("ImageData/" + basename + "/" + note_name):
        os.mkdir("ImageData/" + basename + "/" + note_name)
        os.mkdir("ImageData/" + basename + "/" + note_name + "/Data")
        os.mkdir("ImageData/" + basename + "/" + note_name + "/Visible_Image")
    flow_list = glob.glob(note_path + "/*")
    for flow_path in flow_list:
        flow_name = os.path.basename(flow_path)
        if not os.path.exists("ImageData/" + basename + "/" + note_name + "/Data/" + flow_name):
            os.mkdir("ImageData/" + basename + "/" + note_name + "/Data/" + flow_name)
        data, size = data_list(flow_path + "/*", SAMPLING_FREQUENCY, note_name, flow_name, n_mels=NUMBER_MEL)
        plt.figure()
        librosa.display.specshow(data, sr=SAMPLING_FREQUENCY, x_axis='time')
        plt.colorbar()
        plt.savefig("ImageData/" + basename + "/" + note_name + "/Visible_Image/" + flow_name + ".png")




