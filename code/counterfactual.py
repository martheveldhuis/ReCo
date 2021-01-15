import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CounterfactualGenerator:
    predictor = None
    candidates = None
    scaler = None
    counterfactuals = None
    
    def __init__(self, predictor, scaler):
        self.predictor = predictor
        # Add the predictions to the input variables
        self.candidates = predictor.X_train.assign(NOC = predictor.y_pred)
        # Create a scaled version for distance measurements.
        self.scaler = scaler
        #self.scaled_candidates[self.scaled_candidates.columns] = scaler.fit_transform(self.scaled_candidates[self.scaled_candidates.columns])

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
                self.plot_counterfactual(diff, current_X.name, current_y, target)

        # candidate_scores.sort_values(by=['target_score', 'distance_score', 
        #                                 'features_changed'], inplace = True)
        # print(candidate_scores)
        # best_candidate = candidate_scores.iloc[0]
        # print("best: ", self.predictor.X_train.loc[best_candidate.name,:])

    def calculate_target_score(self, candidate, target):
        return abs(target - candidate)

    def calculate_distance_score(self, candidate, current_X):
        scaled_candidate = self.scaler.transform(candidate.values.reshape(1,-1))
        scaled_current_X = self.scaler.transform(current_X.values.reshape(1,-1))
        
        score = 0
        for i in range(len(scaled_candidate[0])):
            can_val = scaled_candidate[0, i]
            cur_val = scaled_current_X[0, i]
            score += (abs(can_val - cur_val))
        return score

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

        change = current_X.compare(counterfactual)
        
        diff_column = (change["other"] - change["self"])
        change["difference"] = diff_column
        
        return change

    def plot_counterfactual(self, counterfactual, current_name, current_y, target):
        
        fig, ax = plt.subplots(nrows=1, ncols=len(counterfactual.index), 
                               figsize=(15, 5), tight_layout=True)
        fig.suptitle("The profile " + current_name + 
                     " was classified as having {}".format(int(current_y)) + 
                     " contributors. The next best prediction for this profile is {}".format(int(target)) +
                     " contributors. \n This profile could have been predicted to have {}".format(int(target)) + 
                     " contributors if the following features were different (while keeping the other features the same).")
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(len(counterfactual.index)):
            feature = counterfactual.iloc[i]
            feature_name = feature.name
            curr_val = feature.self
            diff = feature.difference

            if diff > 0: # Original sample has lower values than counterfactual.
                bottom = ax[i].bar(feature_name, curr_val, color='tab:gray', 
                                   edgecolor='tab:gray')
                top = ax[i].bar(feature_name, diff, bottom=curr_val, color='w',
                                hatch='.', edgecolor='tab:green')
                ax[i].text(top[0].get_x() + top[0].get_width()/2., 
                           1.04*(top[0].get_height() + bottom[0].get_height()), 
                           '+{:.2f}'.format(diff), ha='center', va='top')
            else: # Original sample has higher values than counterfactual.
                top = ax[i].bar(feature_name, curr_val, color='tab:gray', 
                                edgecolor='tab:gray')
                ax[i].bar(feature_name, abs(diff), bottom=curr_val+diff, 
                          color='tab:gray', hatch ='\\', edgecolor ='tab:red')
                ax[i].text(top[0].get_x() + top[0].get_width()/2., 
                           1.04*(top[0].get_height()), '{:.2f}'.format(diff),
                           ha='center', va='top')

        plt.savefig("counterfactual_v1.png", facecolor=fig.get_facecolor())
