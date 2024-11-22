import os
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
import pandas as pd
from dataset_utils import *
from tqdm import tqdm
# Define the mappings for the data
modality_map = {
    '01': 'full-AV',
    '02': 'video-only',
    '03': 'audio-only'
}

vocal_channel_map = {
    '01': 'speech',
    '02': 'song'
}

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

emotion_to_int = {
    'neutral': 1,
    'calm': 2,
    'happy': 3,
    'sad': 4,
    'angry': 5,
    'fearful': 6,
    'disgust': 7,
    'surprised': 8
}

intensity_map = {
    '01': 'normal',
    '02': 'strong'
}

statement_map = {
    '01': 'Kids are talking by the door',
    '02': 'Dogs are sitting by the door'
}

repetition_map = {
    '01': '1st repetition',
    '02': '2nd repetition'
}


def make_RAVDESS_pd(rav, proc_folder):
    # Here we have an empty array of actors and their recordings
    actors = []

    # We iterate through the directory which has all the actors
    for actor in os.listdir(rav):

        # store another array which has all the files for each actor
        files = os.listdir(os.path.join(rav, actor))

        for file in tqdm(files, desc=actor):
            # part = file.split('.')[0]
            # part = part.split("-")

            # # Destructure the file which is in a format like this "03-02-06-01-02-01-12.wav"
            # modality = modality_map[part[0]
            #                         ] if modality_map[part[0]] else 'unknown'

            # vocal_channel = vocal_channel_map[part[1]
            #                                   ] if vocal_channel_map[part[1]] else 'unknown'

            # Emotion = emotion_map[part[2]
            #                       ] if emotion_map[part[2]] else 'unknown'

            # emotional_intensity = intensity_map[part[3]
            #                                     ] if intensity_map[part[3]] else 'unknown'

            # statement = statement_map[part[4]
            #                           ] if intensity_map[part[4]] else 'unknown'

            # repetition = repetition_map[part[5]
            #                             ] if repetition_map[part[5]] else 'unknown'

            path = (os.path.join(rav, actor, file))
            audio, sr = librosa.load(path, sr=16000)
            audio, _ = librosa.effects.trim(audio, top_db=40)
            audio = librosa.to_mono(audio) 

            num_of_splits = audio.shape[0]//32000

            for i in range(num_of_splits):
                start_idx = i * 32000
                end_idx = min((i + 1) * 32000, audio.shape[0])
                os.makedirs(os.path.join(
                    proc_folder, actor), exist_ok=True)
                sf.write(os.path.join(proc_folder,
                         actor, f"{i}_{file}"), audio[start_idx:end_idx], 16000)

            # actors.append([Emotion, path, modality, vocal_channel,
            #               emotional_intensity, statement, repetition])

    # Save it to a csv file
    # actors_df = pd.DataFrame(actors)
    # actors_df.columns = ['emotion', 'path', 'modality',
    #                      'vocal_channel', 'emotional_intensity', 'statement', 'repetition']
    # actors_df.to_csv(os.path.join(CURRENT_DIR, "DATA", "RAVDESS_MUSIC.csv"))


def DataLoader(df, batch_size, ravdess=False, emotify=False):
    tensors = []
    hop_len = 256
    if ravdess:
        for idx in range(len(df)):
            row = df.iloc[idx]
            path = row['path']
            emotion = row['emotion']

            # Load and preprocess audio
            audio, sr = librosa.load(path, sr=16000)
            audio, _ = librosa.effects.trim(audio, top_db=40)

            # Perform CQT
            C = np.abs(librosa.cqt(audio, sr=16000, hop_length=hop_len))

            # Convert emotion to integer
            # emotion_int = emotion_to_int[emotion]

            # Determine number of chunks if C is longer than 2 seconds (32000 samples)
            chunk_length = sr * 2 // hop_len

            # If C is longer than 2 seconds
            if C.shape[1] > chunk_length:

                # see how many chunks we can get by dividing. // rounds
                num_chunks = C.shape[1] // chunk_length

                for i in range(num_chunks):
                    start_idx = i * chunk_length
                    end_idx = min((i + 1) * chunk_length, C.shape[1])
                    chunk = C[:, start_idx:end_idx]

                    # Ignore the chunk if it is less than 2 seconds
                    if chunk.shape[1] < chunk_length:
                        continue

                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)

                    tensors.append((chunk_tensor, emotion))

            if len(tensors) >= batch_size:
                break

        return tensors


if __name__ == '__main__':
    make_RAVDESS_pd(RAVDESS_MUSIC_DIR, RAVDESS_MUSIC_PROCESSED)
    # df = pd.read_csv(os.path.join(CURRENT_DIR, "DATA", "RAVDESS_MUSIC.csv"))
    # print(df.loc[0])
    # # Example batch size of 4
    # batch = DataLoader(
    #     df, batch_size=4, ravdess=True)
    # print(len(batch))
    # ex, la = batch[0]
