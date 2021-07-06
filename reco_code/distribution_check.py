import json

from scipy.stats import ks_2samp
from data import Dataset
from datareader_19_features import DataReader19Features

# Define which file to read, test fraction, and the custom data reader.
with open('file_paths.json') as json_file:
    file_paths = json.load(json_file)
test_fraction = 0.2

data_reader = DataReader19Features(file_paths['original_590'], test_fraction)
data_reader_samples = DataReader19Features(file_paths['sampled_5000'], test_fraction)

dataset = Dataset(data_reader.read_data())
dataset_samples = Dataset(data_reader_samples.read_data())

# Print KS statistic for the two datasets.
print('total ')
for feature in dataset_samples.feature_names:
    print(feature)
    print(ks_2samp(dataset_samples.data[feature], dataset.data[feature]))

# Print KS statistic for the two datasets, per NOC.
for i in range (1,6):
    print('NOC ' + str(i))
    noc_s = dataset_samples.data[dataset_samples.data['NOC'] == i]
    noc = dataset.data[dataset.data['NOC'] == i]
    for feature in dataset_samples.feature_names:
        print(feature)
        print(ks_2samp(noc_s[feature], noc[feature]))