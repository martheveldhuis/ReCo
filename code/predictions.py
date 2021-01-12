import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from test import DataAnalyzer

class Predictor:

    X_train = None
    y_train = None
    y_pred = None
    model = None

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def define_and_fit(self):
        raise NotImplementedError

    def get_top2_predictions(self, x):
        """
        Paramters:
            x: the input value
        Returns:
            tuple of floats: (first predicted class, secondary class)
        """
        raise NotImplementedError

    def get_instances_predicted_as(self, desired_noc):
        raise NotImplementedError

class RF19Predictor(Predictor):

    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.define_and_fit()

    def define_and_fit(self):
        #BEST_CRITERION = 'entropy' # (default is gini, RFC19 has gini)
        #BEST_N_ESTIMATORS = 100 # (default is 100, RFC19 has 10)
        #BEST_MIN_SAMPLES_SPLIT = 3 # (default is 2, RFC19 has 4)
        #BEST_MAX_LEAF_NODES = None # (default is None, RFC19 has 13)
        #BEST_BOOTSTRAP = True # (default is True, RFC19 has False)
        #BEST_CLASS_WEIGHT = 'balanced' # (default is None, RFC19 has None)

        BEST_CRITERION = 'gini' # (default is gini, RFC19 has gini)
        BEST_N_ESTIMATORS = 10 # (default is 100, RFC19 has 10)
        BEST_MIN_SAMPLES_SPLIT = 4 # (default is 2, RFC19 has 4)
        BEST_MAX_LEAF_NODES = 13 # (default is None, RFC19 has 13)
        BEST_BOOTSTRAP = False # (default is True, RFC19 has False)
        BEST_CLASS_WEIGHT = None # (default is None, RFC19 has None)

        self.model = RandomForestClassifier(n_estimators=BEST_N_ESTIMATORS, 
                                    criterion=BEST_CRITERION,
                                    min_samples_split=BEST_MIN_SAMPLES_SPLIT, 
                                    max_leaf_nodes=BEST_MAX_LEAF_NODES, 
                                    bootstrap=BEST_BOOTSTRAP, 
                                    class_weight=BEST_CLASS_WEIGHT)

        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_train)

    def get_top2_predictions(self, x):
        probs = self.model.predict_proba(x.values.reshape(1, -1))[0]

        highest_prob = np.max(probs)
        top_pred = np.where(probs == highest_prob)[0][0] + 1# convert index to class

        second_highest_prob = 0
        second_pred = None
        for i in range(probs.size):
            if probs[i] > second_highest_prob and probs[i] < highest_prob:
                second_highest_prob = probs[i]
                second_pred = i + 1

        return float(top_pred), float(second_pred)

    def get_instances_predicted_as(self, desired_noc):
        return self.X_train.loc[self.y_pred == desired_noc]


class RF19RPredictor(Predictor):

    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        self.define_and_fit()

    def define_and_fit(self):

        self.model = RandomForestRegressor(bootstrap=True, max_depth=None, 
                                           max_features='sqrt', 
                                           min_samples_leaf=1, 
                                           min_samples_split=2, 
                                           n_estimators=200)

        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_train)

    def get_top2_predictions(self, x):
        
        current_y = self.model.predict(x.values.reshape(1, -1))[0]
        
        rounded_y = current_y.round()
        if rounded_y > current_y: # e.g. 3.8 is rounded to 4.0, 2nd is 3.0
            return rounded_y, rounded_y - 1
        else:                     # e.g. 4.2 is rounded to 4.0, 2nd is 5.0
            return rounded_y, rounded_y + 1

    def get_instances_predicted_as(self, desired_noc):
        return self.X_train.loc[self.y_pred == desired_noc]