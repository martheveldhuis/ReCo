

from data import Dataset
# Our custom implemented data reader behaviour
from datareader_19_features import DataReader19Features
# The predictors that we are going to use
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19
# Anchors explainer
from anchors import AnchorsGenerator
# Counterfactual explainer
from counterfactual import CounterfactualGenerator

# Define which file to read, test fraction, and the custom data reader.
file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590.txt'
test_fraction = 0.2
data_reader = DataReader19Features(file_path, test_fraction)

# Lines below should not have to change.
dataset = Dataset(data_reader.read_data())
# print(dataset.get_features_min_max())
# print(dataset.get_features_mad())
# dataset.plot_feature_correlations()
# dataset.plot_feature_boxplots()
# dataset.plot_feature_histograms()
# dataset.plot_lda()

# Define predictors to use, provide it with a dataset to fit on.
rf_classifier = RFC19(dataset, "RFC19.sav")
rf_regressor = RFR19(dataset, "RFR19.sav")

# Pick the data point you want to have explained.
data_point = dataset.test_data.iloc[0] # 0 (NOC 4, unsure), 4 same ones(NOC 1, sure), 
                                       # 8 is terrible (NOC 5, sure), 9 is interesting (diff reg/class)

# Define Anchors generators (1 generator must be fitted to 1 predictor).
anchors_generator_c = AnchorsGenerator(dataset, rf_classifier)
anchors_generator_r = AnchorsGenerator(dataset, rf_regressor)

# Generate Anchors and print them.
anchor_c = anchors_generator_c.generate_basic_anchor(data_point)
anchor_r = anchors_generator_r.generate_basic_anchor(data_point)
anchor_c.print_anchor_text()
anchor_r.print_anchor_text()

# Define counterfactual generators (1 generator must be fitted to 1 predictor).
CF_generator_c = CounterfactualGenerator(dataset, rf_classifier)
CF_generator_r = CounterfactualGenerator(dataset, rf_regressor)

CF_generator_c.generate_nondominated_train_counterfactuals(data_point)
CF_generator_r.generate_nondominated_train_counterfactuals(data_point)


#####################################OLD#############################
# look at: https://github.com/interpretml/DiCE/blob/f9a92f3cebf857fc589b63b98be56fc42faee904/dice_ml/diverse_counterfactuals.py
