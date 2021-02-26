import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime
# Our dataset object that contains all necessary data information.
from data import Dataset
# Our custom implemented data reader behaviour.
from datareader_19_features import DataReader19Features
# The predictors that we are going to use.
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19
# Anchors explainer.
from anchors import AnchorsGenerator
# Counterfactual explainer.
from counterfactual import CounterfactualGenerator

################################ DATA READING ################################

# Define which file to read, test fraction, and the custom data reader.
file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19.txt'
file_path_samples = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_19.txt'
file_path_merged = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt'
test_fraction = 0.2
data_reader = DataReader19Features(file_path, test_fraction)
data_reader_samples = DataReader19Features(file_path_samples, test_fraction)
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
# rf_classifier = RFC19(dataset, 'RFC19.sav')
# rf_regressor = RFR19(dataset, 'RFR19.sav')
model = RFR19(dataset_merged, 'RFR19_merged.sav')

# Pick the data point you want to have explained.
# data_point = dataset.test_data.iloc[3] # 1 is profile 5B4.3, 3 is profile 2.44
data_point = dataset_merged.test_data.iloc[6] # 0 is profile 5.56, 4 is profile 4.11 (4vs5), 6 still inconsistent.
print(data_point)
print(model.get_prediction(data_point[dataset_merged.feature_names]))
# data_point = dataset.test_data.loc['2.44']


# Define Anchors generators (1 generator must be fitted to 1 predictor).
# anchors_generator_c = AnchorsGenerator(dataset, rf_classifier)

# Generate Anchors and print them.
# anchor = anchors_generator_c.generate_basic_anchor(data_point)
# anchor.plot_anchor()
# anchor.print_anchor_text()

# test_point = data_point.copy()
# test_point = [6.00000000e+00, 9.70000000e+01, 2.00000000e+00, 2.00000000e+00,
#  1.66204986e+00, 4.00000000e+00, 5.00000000e+00, 2.00000000e+00,
#  2.00000000e+00, 6.69784174e-01, 4.00000000e+00, 2.00000000e+00,
#  2.81340979e+03, 9.16143805e-01, 0.00000000e+00, 5.00000000e+00,
#  1.40000000e+01, 8.00000000e-10, 1.00000000e+00]
# test_point = np.array(test_point).reshape(1, -1)
# print('test point is: ')
# print(rf_classifier.get_prediction(test_point))

# cf_data_point = dataset.train_data.loc['2.41']
# anchor_cf = anchors_generator_c.generate_basic_anchor(cf_data_point)
# anchor_cf.print_anchor_text()


# data_point['Loci with 5-6 alleles'] = 2.0
# data_point['Random match probability'] = 0.0002

# print(rf_regressor.get_prediction(data_point[dataset.feature_names]))


# print('Anchors took {}'.format((anchors_end_time-start_time).total_seconds()) + ' to compute')


start_time = datetime.now()

# Define counterfactual generators (1 generator must be fitted to 1 predictor).
# CF_generator_c = CounterfactualGenerator(dataset, rf_classifier)
CF_generator_r = CounterfactualGenerator(dataset_merged, model)

# CF_generator_r.generate_nondominated_train_counterfactuals(data_point)
CF_generator_r.generate_local_avg_train_counterfactual(data_point, n=50)

end_time = datetime.now()
print('Counterfactual took {}'.format((end_time-start_time).total_seconds()) + ' to compute')


# inspiration for code: https://github.com/interpretml/DiCE/blob/f9a92f3cebf857fc589b63b98be56fc42faee904/dice_ml/diverse_counterfactuals.py
# shap.initjs()
# X_train_summary = shap.kmeans(dataset_merged.train_data[dataset_merged.feature_names], 300)


# explainer = shap.KernelExplainer(rf_classifier.get_pred_proba, dataset.train_data[dataset.feature_names])
# explainer = shap.KernelExplainer(rf_regressor.get_prediction, dataset.train_data[dataset.feature_names])
# explainer = shap.KernelExplainer(model.get_prediction, X_train_summary)

# shap_values = explainer.shap_values(cf_data_point[dataset_merged.feature_names])
# f=shap.force_plot(explainer.expected_value, shap_values, cf_data_point[dataset_merged.feature_names], show=False)
# shap.save_html("index3.html", f)

# shap.force_plot(explainer.expected_value[0], shap_values[0], data_point[dataset.feature_names])
# shap.force_plot(explainer.expected_value, shap_values, data_point[dataset.feature_names])#, show=False, matplotlib=True)
#shap.save_html("index.html", f)
#print(shap_values)
#shap.plots.waterfall(shap_values[0])
# plt.savefig("gg.png",bbox_inches='tight')

# cf = dataset_merged.train_data.loc['3.69']
# shap_values = explainer.shap_values(cf[dataset_merged.feature_names])
# f2=shap.force_plot(explainer.expected_value, shap_values, cf[dataset_merged.feature_names], show=False)
# shap.save_html("index2.html", f2)