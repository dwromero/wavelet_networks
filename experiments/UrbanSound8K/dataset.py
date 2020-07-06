# torch
import torch
import torch.optim
import torch.utils.data
import torchaudio
# built-in
import pandas as pd


class UrbanSoundDataset(torch.utils.data.Dataset):
    # rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])

        self.file_path = file_path
        self.folderList = folderList

        self.sample_freq = 22050
        print("Sampling frequency :", self.sample_freq)

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        sample_freq = self.sample_freq

        soundData, sr = torchaudio.load(path, normalization=True)
        soundData = torch.mean(soundData, dim=0, keepdim=True)  # To mono by averaging over the channel axis
        soundData = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_freq)(soundData)  # Resample

        # pad if necessary
        sample_length = sample_freq * 4
        if soundData.shape[-1] != sample_length:
            oldDim = soundData.shape[-1]
            pad_size = (sample_length - oldDim) // 2
            if oldDim % 2 == 0:
                soundData = torch.nn.functional.pad(soundData, (pad_size, pad_size), "constant", 0)
            else:
                soundData = torch.nn.functional.pad(soundData, (pad_size, pad_size + 1), "constant", 0)

        return soundData, self.labels[index]

    def __len__(self):
        return len(self.file_names)


def get_dataset(batch_size, num_workers):
    # Load dataset
    csv_path = 'data/metadata/UrbanSound8K.csv'
    file_path = 'data/audio/'

    train_set = UrbanSoundDataset(csv_path, file_path, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    val_set = UrbanSoundDataset(csv_path, file_path, [10])
    test_set = UrbanSoundDataset(csv_path, file_path, [10])
    print("Train set size: " + str(len(train_set)))
    print("Val set size: " + str(len(val_set)))
    print("Test set size: " + str(len(test_set)))

    # Create TensorDataset and DataLoader objects
    kwargs = {'num_workers': num_workers, 'pin_memory': True} #needed for using datasets on gpu
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    dataloaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
                   'validation': torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle = False, **kwargs)}

    return dataloaders, test_loader
