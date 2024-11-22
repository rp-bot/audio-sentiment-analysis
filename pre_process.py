import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm


class RavdessDataset(Dataset):
    def __init__(self, proc_dir):
        self.processed_dir = proc_dir
        self.file_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for actor_folder in os.listdir(self.processed_dir):
            actor_path = os.path.join(self.processed_dir, actor_folder)

            for file in tqdm(os.listdir(actor_path), actor_folder):
                self.file_paths.append(os.path.join(actor_path, file))

                part = file.split('.')[0]
                part = part.split('_')[1]
                part = part.split("-")

                self.labels.append(int(part[2]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=16000)

        cqt_spectrogram = librosa.cqt(audio, sr=sr, hop_length=256)
        cqt_tensor = torch.tensor(np.abs(cqt_spectrogram), dtype=torch.float32)

        return cqt_tensor, label


if __name__ == "__main__":
    # Testing out the dataloader
    ravdess_dataset = RavdessDataset(RAVDESS_MUSIC_PROCESSED)

    data_loader = DataLoader(ravdess_dataset, 4)

    for batch in data_loader:
        inputs, labels = batch
        for i in range(inputs.shape[0]):
            print(inputs[i], int_to_emotion[labels[i].item()])

        break
