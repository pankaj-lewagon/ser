import librosa
import soundfile
import os, glob
import numpy as np

#Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


#Extract features (mfcc, chroma, mel, temp) from a sound file
def extract_features(file_name, mfcc, chroma, mel, temp):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=40).T,
                            axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                         sr=sample_rate).T,
                             axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,
                          axis=0)
            result = np.hstack((result, mel))
        if temp:
            temp = np.mean(librosa.feature.tempogram(y=X, sr=sample_rate).T,
                           axis=0)
            result = np.hstack((result, temp))
    return result


#Load the data and extract features for each sound file from the ravdess data
def load_data(path_to_data,
              data_files='/Actor_*/*.wav',
              observed_emotions=list(emotions.values()),
              mfcc=True,
              chroma=False,
              mel=True,
              temp=True):
    filenames = path_to_data + data_files
    x, y = [], []
    for file in glob.glob(filenames):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_features(file, mfcc, chroma, mel, temp)
        x.append(feature)
        y.append(emotion)
    return x, y


if __name__ == '__main__':
    path_to_data = "/home/iases/code/pankaj-lewagon/ser/raw_data/ravdess_data"
    x, y = load_data(path_to_data,
                     observed_emotions=[
                         'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust'
                     ])
    print(len(y))
    print(np.array(x).shape)
