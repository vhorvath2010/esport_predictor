import csv

from sklearn.preprocessing import StandardScaler

from match import Match
import numpy as np


def generate_match_array(_match_data):
    return [_match_data[1], _match_data[2], _match_data[3], _match_data[4], _match_data[6], _match_data[7],
            _match_data[8], _match_data[9], _match_data[10], _match_data[11], _match_data[12], _match_data[13]]


def generate_match(_match_array):
    return Match(_match_array[0], _match_array[1], _match_array[2], _match_array[3], _match_array[4], _match_array[5],
                 _match_array[6], _match_array[7], _match_array[8], _match_array[9], _match_array[10],
                 _match_array[11])


# Open match data
with open('match_data.csv') as match_data_file:
    match_data_reader = csv.reader(match_data_file)
    # Read in the first row (labels)
    labels = match_data_reader.__next__()
    # Load in all matches and generate match arrays to standardize
    match_arrays = []
    for match_data in match_data_reader:
        match_arrays.append(generate_match_array(match_data))
    # Standardize matches
    scalar = StandardScaler()
    match_arrays = scalar.fit_transform(match_arrays)
    # Load Match objects
    matches = []
    for match_array in match_arrays:
        matches.append(generate_match(match_array))
    # Save training and testing data using numpy and pickle
    np.random.shuffle(matches)
    train_percent = 0.80
    split_index = int(len(matches) * train_percent)
    np_train = np.array(matches[:split_index])
    np_test = np.array(matches[split_index:])
    np.save('train.npy', np_train, allow_pickle=True)
    np.save('test.npy', np_test, allow_pickle=True)
