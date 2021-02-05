import pandas as pd
import numpy as np
from anchor import utils
from anchor import anchor_tabular
from data import Dataset
from predictor import Predictor

class AnchorsGenerator:
    """A class for generating Anchors."""

    def __init__(self, dataset, predictor):
        """Init method

        :param dataset: Dataset instance containing all data information.
        :param predictor: Predictor instance wrapping all predictor functionality.
        """

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError("should provide data as Dataset instance")

        if isinstance(predictor, Predictor):
            self.predictor = predictor
        else:
            raise ValueError("should provide predictor as Predictor instance")

        self.explainer = self.create_explainer()

    def create_explainer(self):

        # Format classnames and training data
        class_names = self.dataset.data[self.dataset.outcome_name].unique().astype(int).astype(str).tolist()
        feature_names = self.dataset.feature_names
        train_data_np = self.dataset.train_data[self.dataset.feature_names].to_numpy()

        # Create the explainer object
        explainer = anchor_tabular.AnchorTabularExplainer(class_names,
                                                          feature_names, 
                                                          train_data_np)

        return explainer

    def generate_basic_anchor(self, data_point):

        # Ensure we only use the features, not the outcome.
        data_point_X = data_point[self.dataset.feature_names]

        # Helper function required for Anchors.
        def anchor_prediction(x):
            return self.predictor.get_prediction(x) # to deal with regression

        # Generate the explanation.
        data_point_np = data_point_X.to_numpy()
        expl = self.explainer.explain_instance(data_point_np, anchor_prediction, threshold=0.9)

        # Get the predictions.
        top_pred, top_prob, second_pred, second_prob = self.predictor.get_top2_predictions(data_point_X)

        # Create the Anchors object.
        anchor = Anchor(data_point_X, top_pred, top_prob, self.predictor.model_name, 
                        expl.names(), expl.precision(), expl.coverage())
        return anchor
    
    def generate_basic_cf_anchor(self, data_point):

        # Ensure we only use the features, not the outcome.
        data_point_X = data_point[self.dataset.feature_names]

        # Helper function required for Anchors (will not actually be used here).
        def anchor_prediction(x):
            return self.predictor.get_prediction(x).round() # to deal with regression

        # Get the predictions.
        top_pred, top_prob, second_pred, second_prob = self.predictor.get_top2_predictions(data_point_X)

        # Generate the explanation for the cf outcome.
        data_point_np = data_point_X.to_numpy()
        expl = self.explainer.explain_instance(data_point_np, anchor_prediction, threshold=0.9, 
                                               desired_label=second_pred)

        
        # Create the Anchors object.
        anchor = Anchor(data_point_X, second_pred, second_prob, self.predictor.model_name, 
                        expl.names(), expl.precision(), expl.coverage())
        return anchor



class Anchor:
    """Simple wrapper class for Anchors"""

    def __init__(self, data_point, pred, prob, model_name, feature_ranges, precision, coverage):
        """Init method

        :param data_point: pandas series of the data point we are explaining with this anchor.
        :param pred: float for the data point prediction.
        :param prob: float for the probability of the data point prediction.
        :param model_name: string representing the model.
        :param feature_ranges: list of the feature ranges associated with an anchor.
        :param precision: float representing the fraction of the instances which will be 
                        predicted the same as the data point when this anchor holds.
        :param coverage: float representing the probability of the anchor applying to 
                        its perturbation space.
        """

        if isinstance(data_point, pd.Series):
            self.data_point = data_point
        else: 
            raise ValueError("should provide data point in a pandas series")

        if type(pred) is float or isinstance(pred, np.float64):
            self.pred = pred
        else:
            raise ValueError("should provide data point prediction as a float")

        if type(prob) is float or isinstance(prob, np.float64):
            self.prob = prob
        else:
            raise ValueError("should provide data point prediction probability as a float")

        if type(model_name) is str:
            self.model_name = model_name
        else:
            raise ValueError("should provide model name as a string")

        if type(feature_ranges) is list:
            self.feature_ranges = feature_ranges
        else:
            raise ValueError("should provide anchor feature ranges as list of strings")
        
        if type(precision) is float or isinstance(precision, np.float64):
            self.precision = precision
        else:
            raise ValueError("should provide anchor precision as a float")

        if type(coverage) is float or isinstance(coverage, np.float64):
            self.coverage = coverage
        else:
            raise ValueError("should provide anchor coverage as a float")

    def print_anchor_text(self):
        """Simple print of anchor"""

        print('\nProfile ' + self.data_point.name + 
              ' was predicted by model ' + self.model_name + 
              ' to have {}'.format(int(round(self.pred))) + # for regression
              ' contributors, with a probability of {:.2f}'.format(self.prob) + '.')
        print('The model will predict {}'.format(int(round(self.pred))) + # for regression
              ' contributors {}'.format(int(self.precision*100)) + '% of the time' +
              ' when ALL the following rules are true: \n%s ' % ' \n'.join(self.feature_ranges))

    def print_cf_anchor_text(self):
        """Simple print of cf anchor"""

        print('\nProfile ' + self.data_point.name + 
              ' had a secondary prediction by model ' + self.model_name + 
              ' to have {}'.format(int(round(self.pred))) + # for regression
              ' contributors, with a probability of {:.2f}'.format(self.prob) + '.')
        print('The model will predict {}'.format(int(round(self.pred))) + # for regression
              ' contributors {}'.format(int(self.precision*100)) + '% of the time' +
              ' when ALL the following rules are true: \n%s ' % ' \n'.join(self.feature_ranges))