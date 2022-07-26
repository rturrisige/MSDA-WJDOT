import os
import argparse
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
import librosa

def computeMFCC(path_to_wav, wav_name, saver_dir, sr=44100, frameWidth=0.025, frameShift=0.010, nMFCC=13):
    """
    Extracts the MFCCs from the audio.

    Arguments
    ---------
    path_to_wav: str
           path of the folder containing the wav files
    wav_name: str
           name of the wav file
    saver_dir: str
           path of the folder in which saving the mfcc
    sr: int > 0)
           sampling rate
    frameWidth: float
           Hamming window size
    frameShift: float
           Hamming window shift
    nMFCC: int > 0
           number of MFCCs to return
    """
    y, _ = librosa.core.load(path_to_wav+wav_name, sr, mono=True)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=nMFCC, hop_length=int(frameShift * sr), n_fft=int(frameWidth * sr))
    out = mfcc
    saver_name = wav_name.replace("wav", "npy")
    if not os.path.exists(saver_dir):
        os.makedirs(saver_dir)
    np.save(saver_dir + saver_name,out)


def two_files_audio_sum(path1, path2, file_sum_name, gain=0, fmt='wav', repeat=False):
    """ Returns the two mixed audio """
    # Read audio files
    s1 = AudioSegment.from_file(path1).set_sample_width(2)
    s2 = AudioSegment.from_file(path2).set_sample_width(2)
    s2_shift = (len(s1) - len(s2)) / 2 if len(s1) > len(s2) and not repeat else 0
    # Normalization
    s2 = s2.apply_gain(s1.dBFS - s2.dBFS)
    # Reducing loudness of noise audio
    s2 = s2 - gain
    if repeat:
        # Append flipped noise signal
        s2 = s2 + s2.reverse()
    s2_adjusted = AudioSegment.silent(duration=len(s1)).overlay(s2, position=s2_shift, loop=repeat)
    audio_sum = s1.overlay(s2_adjusted)
    audio_sum.export(file_sum_name + '.' + fmt, format=fmt)
    return audio_sum.frame_rate, np.array(audio_sum.get_array_of_samples())


def random_noise_segment_selector(folder, file_ext='wav'):
    # Select example
    dir_file_list = glob(folder + '/*.' + file_ext)
    return random.choice(dir_file_list)


def create_mixed_samples(clean_audio_dir, noise_audio_path, dest_dir, file_ext='wav', repeat_noise=True, min_gain=-10, max_gain=10):
    """
    This function generate a mixed-audio sample for each sample in clean_audio_dir.
    
    Arguments
    ---------
    clean_audio_dir: str
            Directory that cointains clean wav files 
    noise_audio_path: str
            Directories that contain noise wav files
    dest_dir:  str
            Path of the directory where the noisy data will be saved
    file_ext: str
            Extension of audio files
    repeat_noise: True or False
            If it is set, noise is looped when it is shorter than clean sample.
    min_gain: float
            Minimum gain to apply to clean speech
    max_gain: float
            Maximum gain to apply to clean speech

    """
    print(noise_audio_path)
    noise_audio_dirs = [d for d in glob(noise_audio_path + '/*') if os.path.isdir(d)]

    for noise_dir in noise_audio_dirs:
        # Create destination directory if not exist
        noise_name = os.path.basename(noise_dir)
        saver_dir = dest_dir + '/' + noise_name + '/'
        if not os.path.isdir(saver_dir):
            os.makedirs(saver_dir)

        clean_speech_list = glob(os.path.join(clean_audio_dir, '*.' + file_ext))
        total_len = 0
        for clean_speech in clean_speech_list:
            noises_to_combine = []
            # check length if we have variable-length samples
            condition = False
            while not condition:
                condition = True
                noise_to_combine = random_noise_segment_selector(noise_dir)
                if noise_to_combine in noises_to_combine:
                    condition = False
            noises_to_combine.append(noise_to_combine)

            # create and save mixed-speech wavs
            for noise_to_combine in noises_to_combine:
                # choose gain value
                gain = random.uniform(min_gain, max_gain)
                # create a mix
                mix_name = os.path.splitext(os.path.basename(clean_speech))[0] # + '_with_' +  os.path.splitext(os.path.basename(noise_to_combine))[0] + '__' + '{:.2f}'.format(gain) + 'dB'
                print(mix_name)
                sr, audio_sum = two_files_audio_sum(clean_speech, noise_to_combine, os.path.join(saver_dir, mix_name), gain=gain,
                                                    fmt='wav', repeat=repeat_noise)
                total_len += len(audio_sum) / sr

        print('Mixed-audio samples generation completed.')
        print('Total length: {:.2f} seconds'.format(total_len))

