import os
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

from predictor import Predictor
from data import Dataset


class RFC19(Predictor):
    """Implementation of Predictor for the RFC19 sklearn model."""

    def __init__(self, dataset, model_file):
        """Init method

        :param dataset: Dataset instance.
        :param model_file: file where the model is, or will be stored.
        """

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError('should provide a Dataset instance')

        if os.path.isfile(model_file):
            self.model = pickle.load(open(model_file, 'rb'))
        else:
            self.model = self.define_and_fit(model_file)

        self.model_name = model_file
        self.set_predictions()

    def define_and_fit(self, model_file):
        """Defines the RF classifier, based on the feature values found in RFC19 model."""

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

        model = RandomForestClassifier(n_estimators=BEST_N_ESTIMATORS, 
                                       criterion=BEST_CRITERION,
                                       min_samples_split=BEST_MIN_SAMPLES_SPLIT, 
                                       max_leaf_nodes=BEST_MAX_LEAF_NODES, 
                                       bootstrap=BEST_BOOTSTRAP, 
                                       class_weight=BEST_CLASS_WEIGHT)

        X_train = self.dataset.train_data[self.dataset.feature_names]
        y_train = self.dataset.train_data[self.dataset.outcome_name]

        # Fit the model on the training data and store the predictions.
        model.fit(X_train, y_train)

        # Save the model so we don't have to keep fitting.
        pickle.dump(model, open(model_file, 'wb'))
        return model

    def set_predictions(self):
        """Predict on the training data, identify the correctly classified data."""
        
        train_X = self.dataset.train_data[self.dataset.feature_names]
        train_y = self.dataset.train_data[self.dataset.outcome_name]

        # Store the training predictions.
        self.train_pred = self.model.predict(train_X)

        # Calculate accuracy score.
        # acc = accuracy_score(train_y.to_numpy(), self.train_pred)

        # Store the correctly predicted data points by selecting the indices where 
        # the prediction matches the ground truth, making a subset of the training data.
        correct_pred = []
        incorrect_pred = []
        for i in range(len(self.train_pred)):
            if self.train_pred[i] == train_y.iloc[i]:
                correct_pred.append(i)
            else:
                incorrect_pred.append(i)
        self.correct_pred = self.dataset.train_data.iloc[correct_pred, :]
        self.incorrect_pred = self.dataset.train_data.iloc[incorrect_pred, :]          

    def get_prediction_proba(self, x):
        """Just a wrapper for the predict proba function."""
        if isinstance(x, pd.Series):
            return self.model.predict_proba(x.values.reshape(1, -1))[0]
        else:
            return self.model.predict_proba(x)

    def get_prediction(self, x):
        """Just a wrapper for the predict function."""
        if isinstance(x, pd.Series):
            return self.model.predict(x.values.reshape(1, -1))[0]
        else:
            return self.model.predict(x)

    def get_second_prediction(self, x):
        """Get the second most likely prediction."""

        # Use predict_proba method to get probabilites
        if isinstance(x, pd.Series):
            probs = self.model.predict_proba(x.values.reshape(1, -1))[0]
        else:
            probs = self.model.predict_proba(x)

        highest_prob = float(np.max(probs))
        top_pred = float(np.where(probs == highest_prob)[0][0] + 1) # convert index to class

        # Find second best prediction
        second_highest_prob = 0
        second_pred = None
        for i in range(probs.size):
            if probs[i] > second_highest_prob and probs[i] < highest_prob:
                second_highest_prob = float(probs[i])
                second_pred = float(i + 1)

        return second_pred

    def get_data_corr_predicted_as(self, noc):
        """Get all data predicted as a certain NOC"""
        
        train_data_corr = self.correct_pred[self.correct_pred[self.dataset.outcome_name] == noc]
        return train_data_corr


class RFR19(Predictor):
    """Implementation of a random forest regressor sklearn model, using 19 features."""

    def __init__(self, dataset, model_file):
        """Init method

        :param dataset: Dataset instance.
        :param model_file: string of the saved model.
        """

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError('should provide a Dataset instance')

        if os.path.isfile(model_file):
            self.model = pickle.load(open(model_file, 'rb'))
        else:
            self.model = self.define_and_fit(model_file)

        self.model_name = model_file
        self.set_predictions()

    def define_and_fit(self, model_file):
        """Defines a RF regressor"""

        # These parameters originate from a quick grid search
        # model = RandomForestRegressor(max_features='sqrt', 
        #                               n_estimators=200)

        model = RandomForestRegressor()

        X_train = self.dataset.train_data[self.dataset.feature_names]
        y_train = self.dataset.train_data[self.dataset.outcome_name]

        # Fit the model on the training data.
        model.fit(X_train, y_train)

        # Save the model so we don't have to keep fitting.
        pickle.dump(model, open(model_file, 'wb'))
        return model

    def set_predictions(self):
        """
        Predict on the training data, identify the correctly classified data.
        For regression, we will call a prediction 'correct' if the rounded answer == groundtruth. 
        
        """
        
        train_X = self.dataset.train_data[self.dataset.feature_names]
        train_y = self.dataset.train_data[self.dataset.outcome_name] # ints

        # Store the training predictions. It is important to keep floats for cf target calculation 
        # e.g. when NOC is 4: 4.1 is closer than 4.4 
        self.train_pred = self.model.predict(train_X) # floats

        # Calculate accuracy score by using the rounded predictions.
        train_pred_rounded = self.train_pred.round()
        #acc = accuracy_score(train_y.to_numpy(), train_pred_rounded)

        # Store the correctly predicted data points by selecting the indices where the
        # (rounded) prediction matches the ground truth, making a subset of the training data.
        correct_pred = []
        incorrect_pred = []
        for i in range(len(train_pred_rounded)):
            if train_pred_rounded[i] == train_y.iloc[i]:
                correct_pred.append(i)
            else:
                incorrect_pred.append(i)
        self.correct_pred = self.dataset.train_data.iloc[correct_pred, :]
        self.incorrect_pred = self.dataset.train_data.iloc[incorrect_pred, :]        
    
    def get_prediction(self, x):
        """Just a wrapper for the predict function."""
        
        if isinstance(x, pd.Series):
            return self.model.predict(x.values.reshape(1, -1))[0]
        else:
            return self.model.predict(x)

    def get_prediction_proba(self, x):

        pred_1 = self.get_prediction(x)
        pred_2 = self.get_secondary_prediction(x)

        proba_2d_array = np.zeros((1, 5))

        for i in range(len(pred_1)):
            pred_1_rounded = int(pred_1[i].round())
            pred_2 = int(pred_2[i])
            proba_1 = 1 - (abs(pred_1[i] - pred_1_rounded))
            proba_2 = 1 - proba_1
            proba_array = np.zeros(5)
            proba_array[pred_1_rounded-1] = proba_1
            proba_array[pred_2-1] = proba_2
            if i == 0:
                proba_2d_array[0] = proba_array
            else:
                proba_2d_array = proba_2d_array.append(proba_array)

        return proba_2d_array

    def get_secondary_prediction(self, x):
        """Get the second most likely prediction."""

        # For regression, we need to determine which two "classes" are most likely. This is done by
        # rounding to the nearest integer.
        current_y = self.get_prediction(x)
        rounded_y = current_y.round()

        if ((rounded_y > current_y) or # e.g. 3.8 is rounded to 4.0, 2nd option is 3.0
            (rounded_y == 5)):         # 5 is always the max TODO: not hard-code this value
            return rounded_y - 1
        else:                          # e.g. 4.2 is rounded to 4.0, 2nd option is 5.0
            return rounded_y + 1       # all other cases, we default to +1
        

    def get_data_corr_predicted_as(self, noc):
        """Get all data predicted as a certain NOC"""
        
        train_data_corr = self.correct_pred[self.correct_pred[self.dataset.outcome_name] == noc]
        return train_data_corr