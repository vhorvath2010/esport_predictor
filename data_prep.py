import csv
from match import Match
import numpy as np


def generate_match(_match_data):
    return Match(int(_match_data[1]), int(_match_data[2]), int(_match_data[3]), int(_match_data[4]),
                 int(_match_data[6]),
                 int(_match_data[7]), int(_match_data[8]), int(_match_data[9]), int(_match_data[10]),
                 int(_match_data[11]),
                 int(_match_data[12]), int(_match_data[13]))


# Open match data
with open('match_data.csv') as match_data_file:
    match_data_reader = csv.reader(match_data_file)
    # Read in the first row (labels)
    labels = match_data_reader.__next__()
    # Load in all matches and generate Match objects
    matches = []
    for match_data in match_data_reader:
        matches.append(generate_match(match_data))
    # Save training and testing data using numpy and pickle
    np.random.shuffle(matches)
    train_percent = 0.80
    split_index = int(len(matches) * train_percent)
    np_train = np.array(matches[:split_index])
    np_test = np.array(matches[split_index:])
    np.save('train.npy', np_train, allow_pickle=True)
    np.save('test.npy', np_train, allow_pickle=True)
