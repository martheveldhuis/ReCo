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
from evaluation import Evaluator

# inspiration for code: https://github.com/interpretml/DiCE/blob/f9a92f3cebf857fc589b63b98be56fc42faee904/dice_ml/diverse_counterfactuals.py

################################ DATA READING ################################

# Define which file to read, test fraction, and the custom data reader.
# file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19.txt'
# file_path_samples = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_19.txt'
file_path_merged = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt'
file_path_new = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_new_19.txt'
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
# data_point = dataset_merged.test_data.loc['Run 1_Trace 1613676513158'] < diff by 2 methods
# Pick the data point you want to have explained.
# test_points = data_reader_merged.read_data(file_path_new)['data']
# data_point = test_points.iloc[0] # 1_2B.Trace#01
# data_point = test_points.iloc[1] # 1_6B.Trace#01
# data_point = dataset_merged.test_data.loc['2A3.3'] # wrong prediction by model!
# data_point = dataset_merged.train_data.loc['2.37']
# data_point = dataset_merged.test_data.loc['2.29']
# print(data_point)
# dp_pred = rf_regressor_merged.get_prediction(data_point[dataset_merged.feature_names])

# ################################ ANCHORS ################################

# Define Anchors generators (1 generator must be fitted to 1 predictor).
# anchors_generator_c = AnchorsGenerator(dataset_merged, model)

# Generate Anchors and print them.
# anchor = anchors_generator_c.generate_basic_anchor(data_point)
# anchor.plot_anchor()
# anchor.print_anchor_text()


# ################################ SHAP ################################

# Compute SHAP values
shap_generator = ShapGenerator(dataset_merged, rf_regressor_merged, 300)
# dp_shap = shap_generator.get_shap_values(data_point)

# # Create the visualization
# data_point_scaled = dataset_merged.scale_data_point(data_point)
# data_point_X = data_point[dataset_merged.feature_names]
# visualization = Visualization(data_point_X, data_point_scaled, dp_pred, dp_shap)

# ################################ COUNTERFACTUALS ################################


# start_time = datetime.now()
# end_time = datetime.now()
# print('Counterfactual took {}'.format((end_time-start_time).total_seconds()) + ' to compute')

# Define counterfactual generators (1 generator must be fitted to 1
CF_generator = CounterfactualGenerator(dataset_merged, rf_regressor_merged)

# Get the user's input for which NOC to generate the counterfactual.
# def get_user_cf_target(dp_pred):
#     while True:
#         try:
#             cf_target = float(int(input('Enter the NOC explanation you want to see for this profile (1-5): ')))
#         except ValueError:
#             print('You have entered an invalid NOC. Try a whole number between 1 and 5.')
#         else:
#             if cf_target == dp_pred.round():
#                 print('You have entered the same NOC as the current prediction. Try another NOC.')
#                 continue
#             return cf_target



# cf_target = get_user_cf_target(dp_pred)

# cf, cf_scaled, cf_pred, changes = CF_generator.generate_weighted_counterfactual(data_point, data_point_scaled, cf_target)
# visualization.plot_counterfactual(cf, cf_scaled, cf_pred, changes)


# Add tolerance before plotting
# cf_shap = shap_generator.get_shap_values(cf)
# changes = CF_generator.add_shap_tolerance(data_point, dp_shap, dp_pred, cf, cf_shap, cf_target, changes)
# visualization.plot_counterfactual_tol(cf, cf_scaled, cf_pred, changes)


################################ EVALUATION ################################



evaluator = Evaluator(dataset_merged, rf_regressor_merged, CF_generator, shap_generator)


# start_time = datetime.now()
# evaluator.evaluate('dice_genetic')
# end_time = datetime.now()
# print('took {}'.format((end_time-start_time).total_seconds()/60) + ' to compute')

evaluator.plot_boxplot()