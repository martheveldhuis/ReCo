from data import DataReader19Features
from predictions import RF19Predictor
from predictions import RF19RPredictor
from counterfactual import CounterfactualGenerator
from collections import Counter

file_path = r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590.txt"
data_reader = DataReader19Features(file_path, 0.2)

X_train, X_test, y_train, y_test = data_reader.get_split_data()
print("training data X: ", X_train.shape, "y: ", y_train.shape)
print("testing data X: ", X_test.shape, "y: ", y_test.shape)
#print("training data has these counts per class: ", Counter(y_train))
#print("testing data has these counts per class: ", Counter(y_test))

predictor = RF19Predictor(X_train, y_train)
regressor = RF19RPredictor(X_train, y_train)

counterfactual_generator_reg = CounterfactualGenerator(regressor)
counterfactual_generator_class = CounterfactualGenerator(predictor)

current_X = X_test.iloc[0]
current_y_reg, target_y_reg = regressor.get_top2_predictions(current_X)
print("Generating counterfactuals for :", current_X.name)
print("With prediction :", current_y_reg, 
      "And secondary prediction :", target_y_reg)
counterfactual_generator_reg.calculate_counterfactuals(current_X, 
                                                       current_y_reg, 
                                                       target_y_reg)

current_y_class, target_y_class = predictor.get_top2_predictions(current_X)
print("Generating counterfactuals for profile :", current_X.name)
print("With prediction :", current_y_class, 
      "And secondary prediction :", target_y_class)
counterfactual_generator_class.calculate_counterfactuals(current_X, 
                                                         current_y_class, 
                                                         target_y_class)                                                   

data_reader.plot_instance(current_X.name)
data_reader.plot_instance("5.02")