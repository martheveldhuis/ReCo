import pandas as pd
import numpy as np

from data import Dataset
from predictor import Predictor

class CounterfactualGenerator:
    """A class for generating counterfactuals."""

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
    

    def get_non_dominated(self, scores):
        """
        Find the non dominated training points (Pareto efficient).
        :param scores: An (n_points, n_scores) array
        :return: A string array of non dominated profile names.
        """
        non_dom_profiles = scores[:, 2].copy()
        num_scores = np.array(scores[:, 0:2])
        is_efficient = np.ones(scores.shape[0], dtype = bool)
        for i, c in enumerate(num_scores):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(num_scores[is_efficient]<c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        
        return non_dom_profiles[is_efficient]


    def generate_pareto_counterfactual(self, data_point, data_point_scaled, cf_pred):
        """Generate a counterfactual by calculating 2 scores and finding the Pareto optimum one.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param data_point_scaled: a pandas Series object for the scaled instance we want to explain.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index

        # Create arrays to store scores.
        cand_distance = []
        cand_features = []
        profiles = []

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]

            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            
            cand_distance.append(distance_score)
            cand_features.append(features_changed)
            profiles.append(candidate)

            i+=1

        # Put scores together with the profile names.
        candidate_scores = np.array([cand_distance, cand_features, profiles]).T

        # Get the non-dominated profiles and pick the median.
        non_dom = self.get_non_dominated(candidate_scores)
        sorted_scores = candidate_scores[candidate_scores[:, 1].argsort()]
        non_dom_indices = np.isin(sorted_scores[:, 2], non_dom)
        non_dom_sorted = sorted_scores[non_dom_indices]
        # Get the median of the non-dominated set.
        nr_of_cf_found = len(non_dom_sorted)
        cf = non_dom_sorted[int((nr_of_cf_found-1)/2)][2] # take the top candidate's name

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf].copy()
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)

        return original_cf, scaled_cf, cf_pred, changes


    def generate_weighted_counterfactual(self, data_point, data_point_scaled, cf_pred):
        """Generate a counterfactual by calculating 2 scores and summing them.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param data_point_scaled: a pandas Series object for the scaled instance we want to explain.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index

        # Create arrays to store scores.
        cand_distance = []
        cand_features = []
        cand_sum = []
        profiles = []

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]

            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            
            cand_features.append(features_changed)
            cand_distance.append(distance_score)
            profiles.append(candidate)

            i+=1

        # Create a weighted sum.
        weighted_distance = 5 * (np.array(cand_distance))
        weighted_features = 1 * (np.array(cand_features))
        cand_sum = np.add(weighted_distance, weighted_features)
        
        # Put scores together with the profile names.
        candidate_scores = np.array([cand_distance, cand_features, cand_sum, profiles]).T
        sorted_scores = candidate_scores[candidate_scores[:, 2].argsort()] # sort based on sum
        cf = sorted_scores[0][3] # take the top candidate's name

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf].copy()
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)

        return original_cf, scaled_cf, cf_pred, changes 

    def generate_whatif_counterfactual(self, data_point, data_point_scaled, cf_pred):
        """Generate a counterfactual by calculating the distance score only.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param data_point_scaled: a pandas Series object for the scaled instance we want to explain.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index

        # Create arrays to store scores.
        cand_distance = []
        profiles = []

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]

            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            
            cand_distance.append(distance_score)
            profiles.append(candidate)

            i+=1
        
        # Put the score together with the profile names.
        candidate_scores = np.array([cand_distance, profiles]).T
        sorted_scores = candidate_scores[candidate_scores[:, 0].argsort()] # sort based on distance
        cf = sorted_scores[0][1] # take the top candidate's name

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf].copy()
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)
        
        return original_cf, scaled_cf, cf_pred, changes 

    def calculate_distance_score(self, candidate, data_point):
        """Distance score between the original data point and the counterfactual (cf) candidate.
           We separate the distance & nr. of features changed so we can evaluate those separately.
           We are using the range-scaled score: eq 1 in Dandl et al. https://arxiv.org/pdf/2004.11165.pdf

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.

        :returns two floats
        """
        feat_min_max = self.dataset.get_features_min_max()
        feat_mad = self.dataset.get_features_mad()
        total_num_features = len(feat_min_max.columns)
        total_dist = 0
        total_features_changed = 0
        # Get feature-wise distances, scaled by the min-max range of each feature. Keep track of 
        # the number of features that are changed.        
        for feature in feat_min_max.columns:
            dist = abs(candidate[feature] - data_point[feature])
            feature_range = feat_min_max[feature].loc['max'] - feat_min_max[feature].loc['min']
            if dist > 0.0:
                max_scaled_dist = dist / feature_range
                total_dist += max_scaled_dist
                total_features_changed += 1
        # Divide by the total num of features to get values 0-1
        total_dist /= total_num_features
        total_features_changed /= total_num_features
        return total_dist, total_features_changed

    def calculate_scaled_changes(self, counterfactual_scaled, data_point_scaled):
        """Calculate the scaled changes between data point and counterfactual.
        :param counterfactual_scaled: Series of the scaled counterfactual
        :param data_point_scaled: Series of the scaled data point.
        
        """
        # Rename to match data point (so it doesn't come up as a difference).
        counterfactual = counterfactual_scaled.copy()
        data_point = data_point_scaled.copy()
        counterfactual.name = ''
        data_point.name = ''

        # Only keep features that are different.
        compare = data_point.compare(counterfactual)
        
        # Add column to show how the data point would need to change to become the counterfactual.
        diff_column = (compare["other"] - compare["self"])
        compare["difference"] = diff_column

        return compare

    def add_shap_tolerance(self, dp, dp_shap, dp_pred, cf, cf_shap, cf_target, changes):

        # Get the change in SHAP values going from the data point to the cf (pos is increase).
        shap_diff = np.subtract(cf_shap, dp_shap)

        # Grab indices of the changed features to select the relevant shap changes
        changed_feature_indices = np.where(np.isin(self.dataset.feature_names, changes.index))
        changed_shap_dp_to_cf = np.take(shap_diff, changed_feature_indices[0])

        # Add the SHAP changes to a new changes df.
        changes_sorted = changes.copy()
        changes_sorted['shap_changes'] = changed_shap_dp_to_cf

        increase_shap = False
        if cf_target > dp_pred: # Sort by most negative shap change first
            changes_sorted.sort_values(by='shap_changes', inplace=True, ascending=True)
            increase_shap = True
        else:                   # Sort by most postitive shap change first
            changes_sorted.sort_values(by='shap_changes', inplace=True, ascending=False)

        # Iteratively remove changes by deleting the most irrelevant change first.
        features_to_drop = []
        
        for feature in changes_sorted.index:
            # Check if the SHAP value change is already corresponding into the right direction for 
            # at least 0.05. In this way, we won't drop too many changes.
            if increase_shap:
                if changes_sorted.iloc[0]['shap_changes'] > 0.05:
                    break
            elif changes_sorted.iloc[0]['shap_changes'] < -0.05:
                break

            # Check that dropping the change still results in the right prediction.
            changes_sorted.drop(changes_sorted.iloc[0].name, inplace=True)
            new_dp = dp.copy()
            new_dp[changes_sorted.index] = cf[changes_sorted.index]
            new_pred = self.predictor.get_prediction(new_dp[self.dataset.feature_names])

            if new_pred.round() != cf_target:
                break
            else:
                features_to_drop.append(feature)

        return changes.drop(features_to_drop, inplace=False)