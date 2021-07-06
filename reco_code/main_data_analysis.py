import json

from numpy.lib.arraysetops import isin

from data import Dataset
from datareader_19_features import DataReader19Features

# Define which file to read, test fraction, and the custom data reader.
test_fraction = 0.2
with open('file_paths.json') as json_file:
    file_paths = json.load(json_file)
data_reader = DataReader19Features(file_paths['original_590'], test_fraction)
data_reader_samples = DataReader19Features(file_paths['sampled_5000'], test_fraction)
data_reader_merged = DataReader19Features(file_paths['merged_5590'], test_fraction)

# Get stats of 590 dataset
dataset = Dataset(data_reader.read_data())
print(dataset.data.shape)
print(dataset.get_features_min_max())
print(dataset.get_features_mad())
dataset.plot_feature_correlations()
dataset.plot_feature_violin()
dataset.plot_feature_boxplots()
dataset.plot_feature_histograms()
dataset.plot_lda_3d()
dataset.plot_lda_2d()

# Get stats of 5000 sampled dataset
dataset_samples = Dataset(data_reader_samples.read_data())
print(dataset_samples.data.shape)
print(dataset_samples.get_features_min_max())
print(dataset_samples.get_features_mad())
dataset_samples.plot_feature_correlations()
dataset_samples.plot_feature_violin()
dataset_samples.plot_feature_boxplots()
dataset_samples.plot_feature_histograms()
dataset_samples.plot_lda_3d()
dataset_samples.plot_lda_2d()

# Get stats of 5590 merged dataset
dataset_merged = Dataset(data_reader_merged.read_data())
print(dataset_merged.data.shape)
print(dataset_merged.get_features_min_max())
print(dataset_merged.get_features_mad())
dataset_merged.plot_feature_violin()
dataset_merged.plot_feature_correlations()
dataset_merged.plot_feature_boxplots()
dataset_merged.plot_feature_histograms()
dataset_merged.plot_lda_3d()
dataset_merged.plot_lda_2d()