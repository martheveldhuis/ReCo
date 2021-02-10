import numpy as np
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

# Define which file to read, test fraction, and the custom data reader.
file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590.txt'
test_fraction = 0.2
data_reader = DataReader19Features(file_path, test_fraction)

# Lines below should not have to change.
dataset = Dataset(data_reader.read_data())
# print(dataset.get_features_min_max())
# print(dataset.get_features_mad())
# dataset.plot_feature_violin()
# dataset.plot_feature_correlations()
# dataset.plot_feature_boxplots()
# dataset.plot_feature_histograms()
# dataset.plot_lda()

# Define predictors to use, provide it with a dataset to fit on.
rf_classifier = RFC19(dataset, 'RFC19.sav')
#rf_regressor = RFR19(dataset, 'RFR19.sav')

# Pick the data point you want to have explained.
data_point = dataset.test_data.loc['1B3.3']
# data_point = dataset.test_data.loc['2.44']

start_time = datetime.now()

# Define Anchors generators (1 generator must be fitted to 1 predictor).
anchors_generator_c = AnchorsGenerator(dataset, rf_classifier)
# anchors_generator_r = AnchorsGenerator(dataset, rf_regressor)

# Generate Anchors and print them.
anchor = anchors_generator_c.generate_basic_anchor(data_point)
anchor.plot_anchor()
anchor.print_anchor_text()

# test_point = data_point.copy()
# test_point = [6.00000000e+00, 9.70000000e+01, 2.00000000e+00, 2.00000000e+00,
#  1.66204986e+00, 4.00000000e+00, 5.00000000e+00, 2.00000000e+00,
#  2.00000000e+00, 6.69784174e-01, 4.00000000e+00, 2.00000000e+00,
#  2.81340979e+03, 9.16143805e-01, 0.00000000e+00, 5.00000000e+00,
#  1.40000000e+01, 8.00000000e-10, 1.00000000e+00]
# test_point = np.array(test_point).reshape(1, -1)
# print('test point is: ')
# print(rf_classifier.get_prediction(test_point))

# cf_data_point = dataset.train_data.loc['1B2.3']
# anchor_cf = anchors_generator_c.generate_basic_anchor(cf_data_point)
# anchor_cf.print_anchor_text()



# data_point['vWa peak height std.'] = 5000
# data_point['Allele count std.'] = 0.79

# p1, pr1, p2, pr2 = rf_classifier.get_top2_predictions(data_point[dataset.feature_names])
# print(p1)
# print(pr1)

anchors_end_time = datetime.now()
print('Anchors took {}'.format((anchors_end_time-start_time).total_seconds()) + ' to compute')

# Define counterfactual generators (1 generator must be fitted to 1 predictor).
CF_generator_c = CounterfactualGenerator(dataset, rf_classifier)
# CF_generator_r = CounterfactualGenerator(dataset, rf_regressor)

CF_generator_c.generate_nondominated_train_counterfactuals(data_point)
# CF_generator_r.generate_nondominated_train_counterfactuals(data_point)

# cf_end_time = datetime.now()
# print('Counterfactual took {}'.format((cf_end_time-anchors_end_time).total_seconds()) + ' to compute')


# inspiration for code: https://github.com/interpretml/DiCE/blob/f9a92f3cebf857fc589b63b98be56fc42faee904/dice_ml/diverse_counterfactuals.py