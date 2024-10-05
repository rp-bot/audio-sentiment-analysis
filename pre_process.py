import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

# We have the dataset directory here
DATASET_DIR = "P:\ML_datasets\RAVDESS_MUSIC"

# We find the required file here (iteration method)
# for root, dirs, files in os.walk(DATASET_DIR, topdown=False):
#     for name in files:
#         print(os.path.join(root, name))


# But for now we will just work with one directory
Actor_01 = os.path.join(DATASET_DIR, os.listdir(DATASET_DIR)[0])
file_1 = os.path.join(Actor_01, os.listdir(Actor_01)[0])

# We load and get the Constant-Q Transform
y, sr = librosa.load(file_1)
C = np.abs(librosa.cqt(y, sr=sr))

# This is for plotting
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(
#     C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
# ax.set_title('CQT')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
# plt.show()
