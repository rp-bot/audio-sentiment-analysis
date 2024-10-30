import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from dataset_paths import *

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


def make_RAVDESS_pd(rav):
    actors = []

    for actor in os.listdir(rav):

        files = os.listdir(os.path.join(rav, actor))

        for file in files:
            part = file.split('.')[0]
            part = part.split("-")

            modality = modality_map[part[0]
                                    ] if modality_map[part[0]] else 'unknown'

            vocal_channel = vocal_channel_map[part[1]
                                              ] if vocal_channel_map[part[1]] else 'unknown'

            Emotion = emotion_map[part[2]
                                  ] if emotion_map[part[2]] else 'unknown'

            emotional_intensity = intensity_map[part[3]
                                                ] if intensity_map[part[3]] else 'unknown'

            statement = statement_map[part[4]
                                      ] if intensity_map[part[4]] else 'unknown'

            repetition = repetition_map[part[5]
                                        ] if repetition_map[part[5]] else 'unknown'

            try:
                if int(part[6]) % 2 == 0:
                    actor = "female"
                else:
                    # path = (os.path.join(rav, actor, file))
                    # emotion = 'male_'+emotion
                    actor = "male"
            except TypeError:
                actor = "unknown"

            path = (os.path.join(rav, actor, file))
            actors.append([Emotion, path, modality, vocal_channel,
                          emotional_intensity, statement, repetition])

    actors_df = pd.DataFrame(actors)
    actors_df.columns = ['emotion', 'path', 'modality',
                         'vocal_channel', 'emotional_intensity', 'statement', 'repetition']
    actors_df.to_csv(os.path.join(CURRENT_DIR, "DATA", "RAVDESS_MUSIC.csv"))
    print('RAVDESS datasets')
    # print(RavFemales_df.head())


class RAVDESSSongAudio(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio file
        file_path = self.file_paths[idx]
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        # Apply transformation if specified
        if self.transform:
            audio = self.transform(audio)

        # Get the corresponding label
        label = self.labels[idx]
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    make_RAVDESS_pd(RAVDESS_MUSIC_DIR)
