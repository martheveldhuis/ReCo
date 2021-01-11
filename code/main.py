from data import DataReader19Features
from predictions import RF19Predictor
from counterfactual import CounterfactualGenerator
from collections import Counter

file_path = r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590.txt"
data_reader = DataReader19Features(file_path, 0.2)

X_train, X_test, y_train, y_test = data_reader.get_split_data()
print("training data X of shape: ", X_train.shape)
print("training data y of shape: ", y_train.shape)
print("testing data X of shape: ", X_test.shape)
print("testing data y of shape: ", y_test.shape)
print("training data has these counts per class: ", Counter(y_train))
print("testing data has these counts per class: ", Counter(y_test))

predictor = RF19Predictor(X_train, y_train)

current_X = X_test.iloc[0]
current_y = predictor.get_new_prediction(X_test.iloc[0])
print("Generating counterfactuals for :", current_X)
print("With prediction :", current_y)

counterfactual_generator = CounterfactualGenerator(predictor)
counterfactual_generator.calculate_counterfactuals(current_X, current_y, 3.0)
data_reader.plot_instance(current_X.name)
data_reader.plot_instance("3.13")
data_reader.plot_instance("3.34")