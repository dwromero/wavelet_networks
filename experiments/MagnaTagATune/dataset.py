import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from experiments.MagnaTagATune.audio_processor import get_segment_from_npy
import experiments.MagnaTagATune.config as config


'''
Load Dataset (divided into train/validate/test sets)
* audio data : saved as segments in npy file
* labels : 50-d labels in csv file
'''
class SampleLevelMTTDataset(Dataset):
    def __init__(self, mode):
        AUDIO_DIR = '/scratch/experiments/MagnaTagATune/data/npy/'
        LIST_OF_TAGS = '/scratch/experiments/MagnaTagATune/50_tags.txt'
        NUM_TAGS = 50

        self.tag_list = open(LIST_OF_TAGS, 'r').read().split('\n')
        self.audio_dir = AUDIO_DIR
        self.num_tags = NUM_TAGS

        print("dataset mode: ", mode)
        if mode == 'train':
            self.annotation_file = Path(config.BASE_DIR) / 'train_50_tags_annotations_final.csv'

        elif mode == 'valid':
            self.annotation_file = Path(config.BASE_DIR) / 'valid_50_tags_annotations_final.csv'

        elif mode == 'test':
            self.annotation_file = Path(config.BASE_DIR) / 'test_50_tags_annotations_final.csv'

        self.annotations_frame = pd.read_csv(self.annotation_file, delimiter='\t')  # df
        self.labels = self.annotations_frame.drop(['clip_id', 'mp3_path'], axis=1)

    # get one segment (==59049 samples) and its 50-d label
    def __getitem__(self, index):
        idx = index // 10
        segment_idx = index % 10

        mp3filename = self.annotations_frame.iloc[idx]['mp3_path'].split('.')[0] + '.npy'
        try:
            segment = get_segment_from_npy(self.audio_dir + mp3filename, segment_idx)
        except:
            new_index = index - 1 if index > 0 else index + 1
            return self.__getitem__(new_index)

        # build label in the order of 50_tags.txt
        label = np.zeros(self.num_tags)
        for i, tag in enumerate(self.tag_list):
            if tag == '':
                continue
            if self.annotations_frame[tag].iloc[idx] == 1:
                label[i] = 1
        label = torch.FloatTensor(label)
        entry = {'audio': segment, 'label': label}
        return entry

    def __len__(self):
        return self.annotations_frame.shape[0] * 10


def get_dataset(batch_size, num_workers):
    # Load dataset
    train_set = SampleLevelMTTDataset('train')
    valid_set = SampleLevelMTTDataset('valid')
    test_set = SampleLevelMTTDataset('test')

    # Create TensorDataset and DataLoader objects
    kwargs = {'num_workers': num_workers, 'pin_memory': True, 'drop_last': True} #needed for using datasets on gpu
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    dataloaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs),
                   'validation': torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, **kwargs)
                   }

    return dataloaders, test_loader
