import json

from numpy.lib.arraysetops import isin

from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFR19
from counterfactual import CounterfactualGenerator
from shap_values import ShapGenerator
from visualization import Visualization
# from evaluation import Evaluator

########################################## DATA READING ##########################################

# Define which file to read, test fraction, and the custom data reader.
test_fraction = 0.2
test_fraction = 0.2
with open('file_paths.json') as json_file:
    file_paths = json.load(json_file)
data_reader = DataReader19Features(file_paths['merged_5590'], test_fraction)
dataset_merged = Dataset(data_reader.read_data())

########################################### PREDICTORS ###########################################

# Define predictors to use, provide it with a dataset to fit on.

# rf_regressor_merged is the default sklearn regression model, trained on the merged dataset of
# 5590 samples.
rf_regressor_merged = RFR19(dataset_merged, 'RFR19_merged.sav')

# Pick the data point you want to have explained + prediction.
data_point = dataset_merged.test_data.loc['2A3.3'] # profile 2A3.3 is a wrong prediction.
dp_pred = rf_regressor_merged.get_prediction(data_point[dataset_merged.feature_names])

############################################## SHAP ##############################################

# Compute SHAP values
shap_generator = ShapGenerator(dataset_merged, rf_regressor_merged, 300)
dp_shap = shap_generator.get_shap_values(data_point)

# Create the visualization
data_point_scaled = dataset_merged.scale_data_point(data_point)
data_point_X = data_point[dataset_merged.feature_names]
visualization = Visualization(data_point_X, data_point_scaled, dp_pred, dp_shap)

######################################## COUNTERFACTUALS ########################################

# Define counterfactual generators (1 generator must be fitted to 1 dataset + predictor.
CF_generator = CounterfactualGenerator(dataset_merged, rf_regressor_merged)

# Get the user's input for which NOC to generate the counterfactual.
def get_user_cf_target(dp_pred):
    while True:
        try:
            cf_target = float(int(input('Enter the NOC explanation you want to see for this profile (1-5): ')))
        except ValueError:
            print('You have entered an invalid NOC. Try a whole number between 1 and 5.')
        else:
            if cf_target == dp_pred.round():
                print('You have entered the same NOC as the current prediction. Try another NOC.')
                continue
            return cf_target

cf_target = get_user_cf_target(dp_pred)

cf, cf_scaled, cf_pred, changes = CF_generator.generate_pareto_counterfactual(data_point, data_point_scaled, cf_target)
visualization.plot_counterfactual(cf, cf_scaled, cf_pred, changes)

# Add filter before plotting
cf_shap = shap_generator.get_shap_values(cf)
changes = CF_generator.add_shap_tolerance(data_point, dp_shap, dp_pred, cf, cf_shap, cf_target, changes)
visualization.plot_counterfactual_tol(cf, cf_scaled, cf_pred, changes)


########################################### EVALUATION ###########################################


# evaluator = Evaluator(dataset_merged, rf_regressor_merged, CF_generator, shap_generator)
# evaluator.evaluate('reco')
# evaluator.plot_boxplot()