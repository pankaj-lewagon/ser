import os, glob
import numpy as np
from extract_feature import extract_feature
from params import path_to_data, data_files


#DataFlair - Emotions in the RAVDESS dataset
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


#DataFlair - Load the data and extract features for each sound file
def load_data(path_to_data, data_files = 'Actor_*/*.wav', test_size=0.2):
    filenames = path_to_data + data_files
    x, y = [], []
    for file in glob.glob(filenames):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return x, y  #train_test_split(np.array(x), y, test_size=test_size, random_state=9)


if __name__ == '__main__':
    path_to_data = '../raw_data/ravdess_data/'
    x, y = load_data(path_to_data)
    print(len(y))
    print(np.array(x).shape)
