import pandas as pd

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

        columns = ['target_score', 'distance_score']
        candidate_scores = pd.DataFrame(columns = columns)

        for index, candidate in self.candidates.iterrows():
            target_score = self.calculate_target_score(candidate['NOC'], target)
            mask = candidate.index.isin(['NOC'])
            distance_score = self.calculate_distance_score(candidate.loc[~mask], current_X)
            
            new_row = pd.Series({'target_score':target_score, 
                                 'distance_score':distance_score},
                                  name=candidate.name)
            candidate_scores = candidate_scores.append(new_row)
            
        candidate_scores.sort_values(by=['target_score', 'distance_score'], 
                                     inplace = True)
        best_candidate = candidate_scores.iloc[0]
        print(self.predictor.X_train.loc[best_candidate.name,:])


    def calculate_target_score(self, candidate, target):
        return abs(target - candidate)

    def calculate_distance_score(self, candidate, current_X):
        return sum(abs(c - t) for c, t in zip(candidate, current_X))


    def dominates(self, obj1, obj2, sign=[-1, -1]):
        dominates = False
        for a, b, sign in zip(obj1, obj2, sign):
            if a * sign > b * sign:
                dominates = True
            # if one of the objectives is dominated, then return False
            elif a * sign < b * sign:
                return False
        return dominates

    def sortNondominated(candidates, k=None, first_front_only=True):
        if k is None:
            k = len(candidates)

        # Use objectives as keys to make python dictionary
        map_fit_ind = defaultdict(list)
        for i, f_value in enumerate(fitness):  # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
            map_fit_ind[f_value].append(i)
        fits = list(map_fit_ind.keys())  # fitness values

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)  # n (The number of people dominate you)
        dominated_fits = defaultdict(list)  # Sp (The people you dominate)

        # Rank first Pareto front
        # *fits* is a iterable list of chromosomes. Each has multiple objectives.
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                # Eventhougn equals or empty list, n & Sp won't be affected
                if dominates(fit_i, fit_j):
                    dominating_fits[fit_j] += 1  
                    dominated_fits[fit_i].append(fit_j)  
                elif dominates(fit_j, fit_i):  
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0: 
                current_front.append(fit_i)

        fronts = [[]]  # The first front
        for fit in current_front:
            fronts[-1].extend(map_fit_ind[fit])
        pareto_sorted = len(fronts[-1])