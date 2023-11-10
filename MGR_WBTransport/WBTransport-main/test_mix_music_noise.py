# Wasserstein Barycenter Transport for Multi-source Domain Adaptation
#
# References
# ----------
# [1] Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
#     IEEE Transactions on speech and audio processing, 10(5), 293-302.
# [2] http://spib.linse.ufsc.br/noise.html
# [3] Turrisi, R., Flamary, R., Rakotomamonjy, A., & Pontil, M. (2020). Multi-source Domain
#     Adaptation via Weighted Joint Distributions Optimal Transport. arXiv preprint arXiv:2006.12938.

import os
import pydub
import librosa
import argparse
import numpy as np
import soundfile as sf

from utils import overlay_signals, extract_features2

np.random.seed(0)

AMPLITUDE = 32767
MUSIC_DURATION = 30 # Following [1]
NOISE_DURATION = 235 # Following [2]

NOISE_PATH = r'C:\Users\SL276123\Documents\Online DaDiL\WBTransport-main\WBTransport-main\noises'
MUSIC_PATH = r'C:\Users\SL276123\Documents\Online DaDiL\WBTransport-main\WBTransport-main\genres_original'
NOISE_TYPES = [
    "buccaneer2",
]
GENRES = [os.listdir(MUSIC_PATH)[0]]


i = 0
for ndomain, noise_type in enumerate(NOISE_TYPES):
    for nclass, genre in enumerate(GENRES):
        gen_dir = os.path.join(MUSIC_PATH, genre)
        filenames = [os.listdir(gen_dir)[0]]
        for filename in filenames:
            try:
                (sig, rate) = librosa.load(os.path.join(gen_dir, filename), mono=True, duration=MUSIC_DURATION)
            except:
                print("Error while reading file {}".format(filename))

            if noise_type is not None:
                (noise, nrate) = librosa.load(os.path.join(NOISE_PATH, noise_type + '.wav'), mono=True, duration=NOISE_DURATION)
                _, sig, rate = overlay_signals(sig1=sig, rate1=rate, sig2=noise, rate2=nrate)
                sf.write('test_blues_buccaneer2.wav', sig, rate)
            i += 1
