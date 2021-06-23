import json

from numpy.lib.arraysetops import isin
from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFC19
from counterfactual import CounterfactualGenerator
from shap_values import ShapGenerator
from visualization import Visualization
from evaluation import Evaluator


########################################## DATA READING ##########################################

# Define which file to read, test fraction, and the custom data reader.
test_fraction = 0.2
with open('file_paths.json') as json_file:
    file_paths = json.load(json_file)
data_reader = DataReader19Features(file_paths['original_590'], test_fraction)
dataset = Dataset(data_reader.read_data())

########################################### PREDICTORS ###########################################

# Define predictors to use, provide it with a dataset to fit on.

# rfc_19 is the RFC19 model incorporated in DNAStatistX, trained on the original 590 samples.
rfc_19 = RFC19(dataset, 'RFC19.sav')

# Pick the data point you want to have explained + prediction.
data_point = dataset.test_data.iloc[0] 
data_point = dataset.train_data.loc['2A3.3'] # profile 2A3.3 is a wrong prediction by both models.
dp_pred = rfc_19.get_prediction(data_point[dataset.feature_names])

############################################## SHAP ##############################################

# Compute SHAP values
shap_generator = ShapGenerator(dataset, rfc_19, 300)
dp_shap = shap_generator.get_shap_values(data_point)
total_shap = dp_shap

# in case of classification, select the SHAP values of the pred.
top_shap_index = int(dp_pred) - 1
dp_shap = dp_shap[top_shap_index]

# Create the visualization
data_point_scaled = dataset.scale_data_point(data_point)
data_point_X = data_point[dataset.feature_names]
# Multiply SHAP values times 5 for classification because they are a lot lower than for regression.
# Otherwise, they are hardly visible in the visualization.
# This is because for classificaiton the SHAP values add up to the probability of the prediction, 
# starting from the base value of 0.2, they can together become a maximum of 0.8 to reach 1.0. For
# regression, starting from the base value of 2.7, they can together become a maximum of 3.3 to
# reach 5.0.
# TODO: make the visualization scale with the values shown, or remove the numbers on the scale to 
# only show relative magnitude & direction.
visualization = Visualization(data_point_X, data_point_scaled, dp_pred, dp_shap * 5)

######################################## COUNTERFACTUALS ########################################

# Define counterfactual generators (1 generator must be fitted to 1 dataset + predictor.
CF_generator = CounterfactualGenerator(dataset, rfc_19)

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

# in case of classification, get the SHAP values of the dp for the target pred.
cf_target_index = int(cf_target) - 1
dp_shap = total_shap[cf_target_index]

# Add filter before plotting
cf_shap = shap_generator.get_shap_values(cf)
changes = CF_generator.add_shap_tolerance(data_point, dp_shap, dp_pred, cf, cf_shap, cf_target, changes)
visualization.plot_counterfactual_tol(cf, cf_scaled, cf_pred, changes)


#####################################@##### EVALUATION ###################@#######################


# evaluator = Evaluator(dataset_merged, rf_regressor_merged, CF_generator, shap_generator)
# evaluator.evaluate('reco')
# evaluator.plot_boxplot()