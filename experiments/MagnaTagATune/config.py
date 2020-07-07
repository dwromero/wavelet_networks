DATA_DIR = '/scratch/experiments/MagnaTagATune/data/'
BASE_DIR = '/scratch/experiments/MagnaTagATune/data/' # data dir for this model
MTT_DIR = 'data/mp3/' # MTT data dir
AUDIO_DIR = 'data/npy/'
ANNOT_FILE = 'data/annotations_final.csv'
LIST_OF_TAGS = '50_tags.txt'

DEVICE_IDS=[0,1]

# audio params 
SR = 22050
NUM_SAMPLES = 59049
NUM_TAGS = 50

# train params 
BATCH_SIZE = 23
LR = 0.01
DROPOUT = 0.5
NUM_EPOCHS = 100
