import pandas as pd
import numpy as np

class CounterfactualGenerator:
    predictor = None
    candidates = None
    counterfactuals = None
    
    def __init__(self, predictor):
        self.predictor = predictor
        # Add the predictions to the input variables
        self.candidates = predictor.X_train.assign(NOC = predictor.y_pred)

    def get_counterfactuals(self, current_X, current_y, target):
        return self.calculate_counterfactuals(current_X, current_y, target)

    def calculate_counterfactuals(self, current_X, current_y, target):

        columns = ['target_score', 'distance_score', 'features_changed']
        candidate_scores = pd.DataFrame(columns = columns)

        for index, candidate in self.candidates.iterrows():
            target_score = self.calculate_target_score(candidate['NOC'], target)
            
            mask = candidate.index.isin(['NOC'])
            candidate_X = candidate.loc[~mask]

            distance_score = self.calculate_distance_score(candidate_X, 
                                                           current_X)
            features_changed = self.calculate_features_changed(candidate_X, 
                                                               current_X)
            
            new_row = pd.Series({'target_score':target_score, 
                                 'distance_score':distance_score, 
                                 'features_changed':features_changed},
                                  name=candidate.name)
            candidate_scores = candidate_scores.append(new_row)
        
        non_dominated = self.get_non_dominated(candidate_scores)
        
        for i in non_dominated:
            cost = candidate_scores.loc[i]
            if cost['target_score'] == 0:
                print("counterfactual option ", i, "with distance from " ,
                    current_X.name, "of: ", cost['distance_score'], "and ", 
                    cost['features_changed'], "features changed")

                diff = self.calculate_differences(self.candidates.loc[i], current_X)
                print("with differences: ", diff)

        #candidate_scores.sort_values(by=['target_score', 'distance_score', 
        #                                 'features_changed'], inplace = True)
        #print(candidate_scores)
        #best_candidate = candidate_scores.iloc[0]
        #print("best: ", self.predictor.X_train.loc[best_candidate.name,:])
        #return best_candidate.name

    def calculate_target_score(self, candidate, target):
        return abs(target - candidate)

    def calculate_distance_score(self, candidate, current_X):
        return sum(abs(c - t) for c, t in zip(candidate, current_X))

    def calculate_features_changed(self, candidate, current_X):
        count = 0
        for c, t in zip(candidate, current_X):
            if c != t:
                count += 1
        return count
    
    def get_non_dominated(self, costs):
        """
        Find the non dominated points
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

    def calculate_differences(self, counterfactual, current_X):
        mask = counterfactual.index.isin(['NOC'])
        counterfactual = counterfactual.loc[~mask]
        counterfactual.rename({counterfactual.name:current_X.name}, inplace=True)

        change = counterfactual.compare(current_X)
        
        diff_column = abs(change["self"] - change["other"])
        change["difference"] = diff_column
        
        return change