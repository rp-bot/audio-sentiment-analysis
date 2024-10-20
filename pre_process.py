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
file_2 = os.path.join(Actor_01, os.listdir(Actor_01)[1])

# We load and get the Constant-Q Transform
y1, sr1 = librosa.load(file_1)

# Trim edits the audio file so the head and the tail below 40db gets trimmed. 
y1, _ = librosa.effects.trim(y1, top_db=40)
y2, sr2 = librosa.load(file_2)
C1 = np.abs(librosa.cqt(y1, sr=sr1))
C2 = np.abs(librosa.cqt(y2, sr=sr2))

# print(librosa.amplitude_to_db(
#     C1[0][0]))
# print(librosa.amplitude_to_db(
#     C1[40][45:120], ref=np.max))

# This is for plotting
def plot_spectogram(spectogram_array, sample_rate):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(
        spectogram_array, ref=np.max), y_axis='cqt_note', sr=sample_rate, ax=ax)
    ax.set_title('CQT')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
