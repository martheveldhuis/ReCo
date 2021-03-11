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
          
    def generate_counterfactual(self, data_point, data_point_scaled, shap_values):
        """Generate a counterfactual by calculating 3 scores and summing them.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param data_point_scaled: a pandas Series object for the scaled instance we want to explain.
           :param shap_values: a list of floats representing the SHAP values for the current prediction.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        data_point_pred = self.predictor.get_prediction(data_point_X)

        # Get the user's input for which NOC to generate the counterfactual.
        while True:
            try:
                cf_pred = float(int(input('Enter the NOC explanation you want to see for this profile (1-5): ')))
            except ValueError:
                print('You have entered an invalid NOC. Try a whole number between 1 and 5.')
            else:
                if cf_pred == data_point_pred.round():
                    print('You have entered the same NOC as the current prediction. Try another NOC.')
                    continue
                break
      
        # Find profiles classified as cf prediction.
        candidates_X = self.predictor.get_data_corr_predicted_as(cf_pred)[feature_names]
        candidates = candidates_X.index
        candidate_predictions = self.predictor.get_prediction(candidates_X)

        # Create dataframe to store scores.
        columns = ['target_score', 'distance_score', 'features_changed']
        candidate_scores = pd.DataFrame(columns = columns)

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]
            candidate_pred = candidate_predictions[i]

            target_score = self.calculate_target_score(candidate_pred, data_point_pred)
            distance_score, features_changed = self.calculate_distance_score(candidate_X, data_point_X)
            #features_changed = self.calculate_features_changed(candidate_X, data_point_X)
            
            # Put together into a series and append to dataframe as as row.
            new_row = pd.Series({'target_score':target_score, 
                                 'distance_score':distance_score, 
                                 'features_changed':features_changed},
                                  name=candidate)
            candidate_scores = candidate_scores.append(new_row)

            i+=1

        # Sum the scores and sort by ascending order.
        candidate_scores['sum'] = candidate_scores['target_score'] + candidate_scores['features_changed'] + 10 * candidate_scores['distance_score']
        candidate_scores.sort_values(by='sum', inplace=True)
        print(candidate_scores.iloc[0])
        cf = candidate_scores.iloc[0]

        # Get the CF profile and calculate the changes.
        original_cf = self.dataset.train_data.loc[cf.name].copy()
        scaled_cf = self.dataset.scale_data_point(original_cf)
        changes = self.calculate_scaled_changes(scaled_cf, data_point_scaled)

        # Check the prediction
        print('new prediction: ')
        print(self.predictor.get_prediction(original_cf[feature_names]))


        return original_cf, scaled_cf, cf_pred, changes


    def calculate_target_score(self, candidate_pred, data_point_pred):
        """Score between counterfactual (cf) candidate pred score and the data point pred.
           e.g. when we have a data point with prediction 3.1 and looking for a cf with target 4,
           we will favor predictions such as 3.6 over ones like 4.3.
           This will most likely show more similar profiles rather than setting the target as 4.
           
        :param candidate_pred: float representing the counterfactual (cf) candidate's prediction.
        :param target_pred: float representing the data point's prediction.
        """

        return abs(data_point_pred - candidate_pred)

    def calculate_distance_score(self, candidate, data_point):
        """Distance score between the original data point and the counterfactual (cf) candidate. 
           We are using the range-scaled score: eq 1 in Dandl et al. https://arxiv.org/pdf/2004.11165.pdf

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.
        """

        num_features_changed = 0
        features_min_max = self.dataset.get_features_min_max()
        score = 0        
        for feature in features_min_max.columns:
            dist = abs(candidate[feature] - data_point[feature])
            max_scaled_dist = dist / (features_min_max[feature].loc['max'])
            if max_scaled_dist > 0.0: # Allow for 5% tolerance
                score += max_scaled_dist
                num_features_changed += 1
        score = score / len(features_min_max.columns)
        changed = num_features_changed / len(features_min_max.columns)
        return score, changed

    # def calculate_features_changed(self, candidate, data_point):
    #     """Count of how many features are changed between the original data point and the 
    #        counterfactual (cf) candidate. Note that even if the value is only slightly off, 
    #        this will count towards this score. 

    #     :param candidate: the counterfactual(cf) candidate's feature values.
    #     :param data_point: the data point's feature values.
    #     """
        
    #     num_features = len(data_point.index)
    #     count = 0
    #     for c, t in zip(candidate, data_point):
    #         if c != t:
    #             count += 1
    #     return count/num_features
    
    def get_non_dominated(self, costs):
        """Find the non dominated profiles based on their 3 cost scores.
        
        :param costs: DataFrame of 3 cost scores per profile.
        :return non_dominated_set: a list of profile names that are non-dominated.
        """

        non_dominated = np.ones(costs.shape[0], dtype = bool)
        non_dominated_set = []

        i = 0     
        for label, c in costs.iterrows():
            non_dominated[i] = (np.all(np.any(costs[:i]>c, axis=1)) and 
                                np.all(np.any(costs[i+1:]>c, axis=1)) and
                                np.all(np.any(costs[i+2:]>c, axis=1)))
            if non_dominated[i] == True:
                non_dominated_set.append(label)
            i+=1
        
        return non_dominated_set

    def calculate_scaled_changes(self, counterfactual_scaled, data_point_scaled):
        """Calculate the pairwise changes between data point and counterfactual."""

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

        # Remove changes that are less than 5%
        #compare = compare[abs(compare['difference']) > 0.05]
        return compare
