import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from test import DataAnalyzer

# Read data
data_analyzer = DataAnalyzer()
data_analyzer.read_file()

# Split in a stratified manner
X_train, X_test, y_train, y_test = train_test_split(data_analyzer.X, 
                                                    data_analyzer.y, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    stratify=data_analyzer.y)
y_train = y_train.values.reshape(-1,1).ravel()
y_test = y_test.values.reshape(-1,1).ravel()
                                                    
print("training set ", X_train.shape)
print("training labels ", y_train.shape)
print("testing labels ", y_test.shape)
print(Counter(y_train))
print(Counter(y_test))

BEST_CRITERION = 'entropy' # (default is gini, RFC19 has gini)
BEST_N_ESTIMATORS = 100 # (default is 100, RFC19 has 10)
BEST_MIN_SAMPLES_SPLIT = 3 # (default is 2, RFC19 has 4)
BEST_MAX_LEAF_NODES = None # (default is None, RFC19 has 13)
BEST_BOOTSTRAP = True # (default is True, RFC19 has False)
BEST_CLASS_WEIGHT = 'balanced' # (default is None, RFC19 has None)

model = RandomForestClassifier(n_estimators=BEST_N_ESTIMATORS, 
                               criterion=BEST_CRITERION,
                               min_samples_split=BEST_MIN_SAMPLES_SPLIT, 
                               max_leaf_nodes=BEST_MAX_LEAF_NODES, 
                               bootstrap=BEST_BOOTSTRAP, 
                               class_weight=BEST_CLASS_WEIGHT)

model.fit(X_train, y_train)


class PredictorWrapperBase:

    def get_prediction(self, x):
        raise NotImplementedError

    def get_instances_predicted_as(self, desired_noc):
        raise NotImplementedError

class PredictorWrapper1(PredictorWrapperBase):
    
    predictor = None

    def __init__(self, predictor): 
        self.predictor = predictor

    def get_prediction(self, x):
        return self.predictor.predict(x)

prediction_wrapper = PredictorWrapper1(model)
print(prediction_wrapper.get_prediction(X_test))