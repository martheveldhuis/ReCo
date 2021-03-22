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

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        data_point_pred = self.predictor.get_prediction(data_point_X)
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index
        candidate_predictions = self.predictor.get_prediction(candidates_X)

        # Create 2D array to store scores.
        cand_distance = []
        cand_features = []
        profiles = []

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]
            candidate_pred = candidate_predictions[i]

            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            
            cand_distance.append(distance_score)
            cand_features.append(features_changed)
            profiles.append(candidate)

            i+=1

        # Get the non-dominated profiles and pick the non-dom features
        candidate_scores = np.array([cand_distance, cand_features, profiles]).T
        non_dom = self.get_non_dominated(candidate_scores)
        sorted_scores = candidate_scores[candidate_scores[:, 1].argsort()]

        non_dom_indices = np.isin(sorted_scores[:, 2], non_dom)
        non_dom_sorted = sorted_scores[non_dom_indices]
        print(non_dom_sorted[0])
        
        cf = non_dom_sorted[0][2]

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

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        data_point_pred = self.predictor.get_prediction(data_point_X)
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index
        candidate_predictions = self.predictor.get_prediction(candidates_X)

        # Create dataframe to store scores.
        columns = ['distance_score', 'features_changed']
        candidate_scores = pd.DataFrame(columns = columns)

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]
            candidate_pred = candidate_predictions[i]

            target_score = self.calculate_target_score(cf_pred, candidate_pred, data_point_pred)
            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            
            # Put together into a series and append to dataframe as as row.
            new_row = pd.Series({'distance_score':distance_score, 
                                 'features_changed':features_changed},
                                  name=candidate)
            candidate_scores = candidate_scores.append(new_row)

            i+=1

        # Sum the scores and sort by ascending order.
        candidate_scores['sum'] = candidate_scores['features_changed'] + 6 * candidate_scores['distance_score']
        candidate_scores.sort_values(by='sum', inplace=True)
        cf = candidate_scores.iloc[0]

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf.name].copy()
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)

        return original_cf, scaled_cf, cf_pred, changes 

    def generate_avg_counterfactual(self, data_point, data_point_scaled, cf_pred):
        """Generate a counterfactual by weighing each feature value by distance.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param data_point_scaled: a pandas Series object for the scaled instance we want to explain.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        data_point_pred = self.predictor.get_prediction(data_point_X)
      
        # Find profiles classified as cf prediction.
        training_data_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        training_data = training_data_X.index
        training_data_predictions = self.predictor.get_prediction(training_data_X)

        # Create dataframe to store scores.
        columns = ['distance_score', 'features_changed']
        training_data_scores = pd.DataFrame(columns = columns)

        # Calculate all scores for each candidate data point.
        i=0
        weighted_cf = pd.DataFrame(1, index=[0], columns=feature_names)
        total_weight = 0

        for training_point in training_data:
            training_point_X = training_data_X.loc[training_point]
            training_point_pred = training_data_predictions[i]

            distance_score, features_changed = self.calculate_distance_score(training_point_X, data_point_X)

            # Put together into a series and append to dataframe as as row.
            new_row = pd.Series({'distance_score': distance_score, 
                                 'features_changed': features_changed},
                                  name=training_point)
            training_data_scores = training_data_scores.append(new_row)

            weight = 10 * (1 - distance_score) + (1 - features_changed)
            total_weight += weight
            weighted_addition = training_point_X.multiply(weight)
            weighted_cf += weighted_addition

            i+=1
       
        weighted_cf /= total_weight
        cf = weighted_cf.iloc[0]
        cf.name = 'avg'
        cf.columns = feature_names
        cf_X = cf[feature_names]

        # Create dataframe to store scores.
        candidate_scores = pd.DataFrame(columns = columns)

        i = 0
        for candidate in training_data:
            candidate_X = training_data_X.loc[candidate]
            candidate_pred = training_data_predictions[i]

            distance_score, features_changed = self.calculate_distance_score(candidate_X, cf_X)
            
            # Put together into a series and append to dataframe as as row.
            new_row = pd.Series({'distance_score': distance_score, 
                                 'features_changed': features_changed},
                                  name=candidate)
            candidate_scores = candidate_scores.append(new_row)

            i+=1

        # Sum the scores and sort by ascending order.
        candidate_scores['sum'] = candidate_scores['features_changed'] + 6 * candidate_scores['distance_score']
        candidate_scores.sort_values(by='sum', inplace=True)
        cf = candidate_scores.iloc[0]

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf.name]
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)

        return original_cf, scaled_cf, cf_pred, changes

    def calculate_target_score(self, cf_pred, candidate_pred, data_point_pred):
        """Score between counterfactual (cf) candidate pred score and the data point pred.
           e.g. when we have a data point with prediction 3.1 and looking for a cf with target 4,
           we will favor predictions such as 3.6 over ones like 4.3.
           This will most likely show more similar profiles rather than setting the target as 4.
        :param cf_pred: int representing the target    
        :param candidate_pred: float representing the counterfactual (cf) candidate's prediction.
        :param data_point_pred: float representing the data point's prediction.
        """            

        return abs(data_point_pred - candidate_pred)

    def calculate_distance_score(self, candidate, data_point):
        """Distance score between the original data point and the counterfactual (cf) candidate.
           We separate the distance & nr. of features changed so we can evaluate those separately.
           We are using the range-scaled score: eq 1 in Dandl et al. https://arxiv.org/pdf/2004.11165.pdf

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.

        :returns two floats
        """
        features_min_max = self.dataset.get_features_min_max()
        total_num_features = len(features_min_max.columns)
        dist = 0
        features_changed = 0
        # Get feature-wise distances, scaled by the max value of each feature. Keep track of the
        # number of features that are changed.        
        for feature in features_min_max.columns:
            dist = abs(candidate[feature] - data_point[feature])
            max_scaled_dist = dist / (features_min_max[feature].loc['max'])
            if max_scaled_dist > 0.0:
                dist += max_scaled_dist
                features_changed += 1
        # Divide by the total num of features to get values 0-1
        dist /= total_num_features
        features_changed /= total_num_features
        return dist, features_changed

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
        shap_dp_to_cf = np.subtract(cf_shap, dp_shap)

        # Grab indices of the changed features to select the relevant shap changes
        changed_feature_indices = np.where(np.isin(self.dataset.feature_names, changes.index))
        changed_shap_dp_to_cf = np.take(shap_dp_to_cf, changed_feature_indices[0])

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
            elif changes_sorted.iloc[0]['shap_changes'] < 0.05:
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



        # i = 0
        # for feature_name in self.dataset.feature_names:
        #     if feature_name in changes.index:

        #         dp_shap_feature = dp_shap[i]
        #         cf_shap_feature = cf_shap[i]
        #         shap_change = cf_shap_feature - dp_shap_feature

        #         # Drop changes where the change in SHAP values does not correspond with the change
        #         # in prediction (or is zero / less than 0.05).
        #         if ((cf_target > dp_pred and shap_change <= min_change) or 
        #             (cf_target < dp_pred and shap_change >= -min_change)): 
        #             changes.drop(feature_name, inplace=True)
        #     i+=1
