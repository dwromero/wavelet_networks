''' Run this file to process raw audio '''
import os, errno
import numpy as np
import torch 
import librosa
from pathlib import Path
import experiments.MagnaTagATune.config as config


def save_audio_to_npy(rawfilepath, npyfilepath):
    ''' Save audio signal with sr=sample_rate to npy file 
    Args : 
        rawfilepath : path to the MTT audio files 
        npyfilepath : path to save the numpy array audio signal 
    Return :
        None 
    '''
    
    # make directory if not existing 
    if not os.path.exists(npyfilepath):
        os.makedirs(npyfilepath)


    mydir = [path for path in os.listdir(rawfilepath) if path >= '0' and path <= 'f']
    for path in mydir : 
        # create directory with names '0' to 'f' if it doesn't already exist
        try:
            os.mkdir(Path(npyfilepath) / path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        audios = [audio for audio in os.listdir(Path(rawfilepath) / path) if audio.split(".")[-1] == 'mp3']
        for audio in audios :
            try:
                y,sr = librosa.load(rawfilepath + path + '/' + audio, sr=config.SR)
                if len(y)/config.NUM_SAMPLES < 10:
                    print ("There are less than 10 segments in this audio")
            except:
                print ("Cannot load audio {}".format(audio))
                continue

            fn = audio.split(".")[0]
            np.save(Path(npyfilepath) / (path + '/' + fn + '.npy'), y)


def get_segment_from_npy(npyfile, segment_idx):
    ''' Return random segment of length num_samples from the audio 
    Args : 
        npyfile : path to all the npy files each containing audio signals 
        segment_idx : index of the segment to retrieve; max(segment_idx) = total_samples//num_samples
    Return : 
        segment : audio signal of length num_samples 
    '''
    song = np.load(npyfile)
    # randidx = np.random.randint(10)
    try : 
        segment = song[segment_idx * config.NUM_SAMPLES : (segment_idx+1)*config.NUM_SAMPLES]
    except : 
        randidx = np.random.randint(10)
        get_segment_from_npy(npyfile, randidx, config.NUM_SAMPLES)
    return segment



if __name__ =='__main__':
    # read audio signal and save to npy format 
    save_audio_to_npy(config.MTT_DIR, config.AUDIO_DIR)
    

