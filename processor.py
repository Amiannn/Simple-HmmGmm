import os
import numpy as np
import speech.frontend as feat

import scipy.io.wavfile as wav

from tqdm import tqdm

def dataloader(path):
    datas, labels = [], []
    for filename in tqdm(os.listdir(path)):
        filepath = os.path.join(path, filename)
        
        # read wav file
        fs_hz, signal = wav.read(filepath)

        # get sound feature and label
        data  = feat.compute_mfcc(signal, fs_hz)
        # data  = feat.compute_fbank(signal, fs_hz)
        label = filename.split('_')[0]

        datas.append(data)
        labels.append(label)
    return datas, labels