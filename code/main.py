import numpy as np
import pandas as pd
import math
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19
from anchors import AnchorsGenerator
from counterfactual import CounterfactualGenerator
from shap_values import ShapGenerator
from visualization import Visualization

# inspiration for code: https://github.com/interpretml/DiCE/blob/f9a92f3cebf857fc589b63b98be56fc42faee904/dice_ml/diverse_counterfactuals.py

################################ DATA READING ################################

# Define which file to read, test fraction, and the custom data reader.
# file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19.txt'
# file_path_samples = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_19.txt'
file_path_merged = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt'
test_fraction = 0.2
# data_reader = DataReader19Features(file_path, test_fraction)
# data_reader_samples = DataReader19Features(file_path_samples, test_fraction)
data_reader_merged = DataReader19Features(file_path_merged, test_fraction)

# Get stats of 590 dataset
# dataset = Dataset(data_reader.read_data())
# print(dataset.data.shape)
# print(dataset.get_features_min_max()
# print(dataset.get_features_mad())
# dataset.plot_feature_violin()
# dataset.plot_feature_correlations()
# dataset.plot_feature_boxplots()
# dataset.plot_feature_histograms()
# dataset.plot_lda_3d()
# dataset.plot_lda_2d()

# Get stats of 5000 sampled dataset
# dataset_samples = Dataset(data_reader_samples.read_data())
# print(dataset_samples.data.shape)
# print(dataset_samples.get_features_min_max())
# print(dataset_samples.get_features_mad())
# dataset_samples.plot_feature_violin()
# dataset_samples.plot_feature_correlations()
# dataset_samples.plot_feature_boxplots()
# dataset_samples.plot_feature_histograms()
# dataset_samples.plot_lda_3d()
# dataset_samples.plot_lda_2d()

# Get stats of 5590 merged dataset
dataset_merged = Dataset(data_reader_merged.read_data())
# print(dataset_merged.data.shape)
# print(dataset_merged.get_features_min_max())
# print(dataset_merged.get_features_mad())
# dataset_merged.plot_feature_violin()
# dataset_merged.plot_feature_correlations()
# dataset_merged.plot_feature_boxplots()
# dataset_merged.plot_feature_histograms()
# dataset_merged.plot_lda_3d()
# dataset_merged.plot_lda_2d()


################################ PREDICTORS ################################


# Define predictors to use, provide it with a dataset to fit on.
rf_regressor_merged = RFR19(dataset_merged, 'RFR19_merged.sav')

# Pick the data point you want to have explained.
data_point = dataset_merged.test_data.loc['2A3.3'] # wrong prediction by model!
# data_point = dataset_merged.train_data.loc['2C3.3']
# data_point = dataset_merged.test_data.loc['5.79']
# data_point = dataset_merged.test_data.iloc[20] # 0 is profile 5.56, 20 is trace x14.
# data_point = dataset_merged.train_data.loc['Run 1_Trace 1613482097080']
print(data_point)

# data_point['TAC'] = 96
# data_point['vWa peak height variation'] = 430.1
# data_point['allele count variation'] = 0.9161
# data_point['Loci with 5-6 alleles'] = 10
print(rf_regressor_merged.get_prediction(data_point[dataset_merged.feature_names]))


################################ ANCHORS ################################

# Define Anchors generators (1 generator must be fitted to 1 predictor).
# anchors_generator_c = AnchorsGenerator(dataset_merged, model)

# Generate Anchors and print them.
# anchor = anchors_generator_c.generate_basic_anchor(data_point)
# anchor.plot_anchor()
# anchor.print_anchor_text()


################################ SHAP ################################

# Compute SHAP values
shap_generator = ShapGenerator(dataset_merged, rf_regressor_merged, 300)
shap_values = shap_generator.get_shap_values(data_point)

# Create the visualization
data_point_scaled = dataset_merged.scale_data_point(data_point)
data_point_X = data_point[dataset_merged.feature_names]
prediction = rf_regressor_merged.get_prediction(data_point_X)
print(prediction)
visualization = Visualization(data_point_X, data_point_scaled, prediction, shap_values)

################################ COUNTERFACTUALS ################################


start_time = datetime.now()

# Define counterfactual generators (1 generator must be fitted to 1 predictor).
CF_generator = CounterfactualGenerator(dataset_merged, rf_regressor_merged)

cf, cf_scaled, cf_pred, changes = CF_generator.generate_counterfactual(data_point, 
                                                                       data_point_scaled,
                                                                       shap_values)


end_time = datetime.now()
print('Counterfactual took {}'.format((end_time-start_time).total_seconds()) + ' to compute')

visualization.plot_counterfactual(cf, cf_scaled, cf_pred, changes)