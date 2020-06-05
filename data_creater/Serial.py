import serial
import sounddevice as sd
from scipy.io.wavfile import write
import time
import sys
import os
import numpy as np

fs = 44100  # サンプリングレート
BIT = 9600
rec = "rec1"
path = None
mic = ["mic1", "mic2"]  # mic1: FIFINE mic2: JTS

def ser_write(py_input):
    with serial.Serial('COM3', BIT, timeout=None) as ser:
        ser.write(py_input.encode())


def ser_read(print_message=False):
    with serial.Serial('COM3', BIT, timeout=None) as ser:
            x = ser.readline().decode()
            if print_message:
                return print(x)
            else:
                return x


def flow_setting():
    print("input min flow value 0.[][]")
    min_flow = float(input())
    print("min flow value : {}".format(min_flow * 0.01))
    a = int(min_flow % 10)
    ser_write(str(a))
    ser_read(True)

    print("input max flow value 0.[][]")
    max_flow = float(input())
    print("max flow value : {}".format(max_flow * 0.01))
    b = int((min_flow - a) // 10)
    ser_write(str(b))
    ser_read(True)
    return min_flow * 0.01, max_flow * 0.01


def recording(min_flow, max_flow):
    while 1:
        print("start recording? : yes no")
        st = input()
        if not((st == "yes") or (st == "no")):
            print("{} was not expected. please input start or exit".format(st))
            continue
        elif st == "exit":
            print("return first")
            return 1
        else:
            break
    print("Recording start")
    rec_number = int((max_flow - min_flow) / 0.002) + 1
    print("Total recording : {}".format(rec_number))
    dev = [1, 3]
    sd.default.channels = 2
    sd.default.device = dev
    sd.default.samplerate = fs

    for i in range(0, rec_number):
        ser_write("a")
        flow = min_flow + i * 0.002
        tmp_path = path + "{:.3f}".format(flow)
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        print("flow : {}".format(min_flow + i * 0.002))
        if not os.path.exists(tmp_path + "/" + mic[0]):
            os.mkdir(tmp_path + "/" + mic[0])
        if not os.path.exists(tmp_path + "/" + mic[1]):
            os.mkdir(tmp_path + "/" + mic[1])

        for n in range(0, 5):
            pro_bar = ('=' * (n+1)) + (' ' * (4 - n))
            print('\rrecording  [{0}] {1}/{2}'.format(pro_bar, n+1, 5), end='')
            time.sleep(2)
            record = sd.rec(int(10 * fs))
            sd.wait()
            write(tmp_path + "/" + mic[0] + "/" + str(n) + ".wav", fs, record[:, 0])
            write(tmp_path + "/" + mic[1] + "/" + str(n) + ".wav", fs, record[:, 1])
        print("")
    ser_write("b")
    ser_read(True)



def start_system():
    print("Recorder Recording System")
    print("input start or exit")
    st = input()
    if not ((st == "exit") or (st == "start")):
        print("{} was not expected. please input start or exit".format(st))
        return 0
    elif st == "exit":
        return 1


while 1:
    start = start_system()  # システムスタート
    if start == 1:
        break
    elif start == 0:
        continue

    if not os.path.exists("Data"):
        os.mkdir("Data")
    if not os.path.exists("Data/" + rec):
        os.mkdir("Data/" + rec)
    path = "Data/" + rec + "/"
    min_flow, max_flow = flow_setting()  # 流量セット
    recording(min_flow, max_flow)  # レコーディング関数