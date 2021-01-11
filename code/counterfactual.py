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
        
        self.get_non_dominated(candidate_scores)

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
        
        print(non_dominated_set)
        for i in non_dominated_set:
            print(costs.loc[i])


    # def dominates(self, obj1, obj2, sign=[-1, -1]):
    #     dominates = False
    #     for a, b, sign in zip(obj1, obj2, sign):
    #         if a * sign > b * sign:
    #             dominates = True
    #         # if one of the objectives is dominated, then return False
    #         elif a * sign < b * sign:
    #             return False
    #     return dominates

    # def sortNondominated(candidates, k=None, first_front_only=True):
    #     if k is None:
    #         k = len(candidates)

    #     # Use objectives as keys to make python dictionary
    #     map_fit_ind = defaultdict(list)
    #     for i, f_value in enumerate(fitness):  # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
    #         map_fit_ind[f_value].append(i)
    #     fits = list(map_fit_ind.keys())  # fitness values

    #     current_front = []
    #     next_front = []
    #     dominating_fits = defaultdict(int)  # n (The number of people dominate you)
    #     dominated_fits = defaultdict(list)  # Sp (The people you dominate)

    #     # Rank first Pareto front
    #     # *fits* is a iterable list of chromosomes. Each has multiple objectives.
    #     for i, fit_i in enumerate(fits):
    #         for fit_j in fits[i + 1:]:
    #             # Eventhougn equals or empty list, n & Sp won't be affected
    #             if dominates(fit_i, fit_j):
    #                 dominating_fits[fit_j] += 1  
    #                 dominated_fits[fit_i].append(fit_j)  
    #             elif dominates(fit_j, fit_i):  
    #                 dominating_fits[fit_i] += 1
    #                 dominated_fits[fit_j].append(fit_i)
    #         if dominating_fits[fit_i] == 0: 
    #             current_front.append(fit_i)

    #     fronts = [[]]  # The first front
    #     for fit in current_front:
    #         fronts[-1].extend(map_fit_ind[fit])
    #     pareto_sorted = len(fronts[-1])