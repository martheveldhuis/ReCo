import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

    def generate_nondominated_train_counterfactuals(self, data_point):
        """Generate counterfactuals by simulataneously optimizing 3 objectives as inspired by
           Dandl et al. https://arxiv.org/pdf/2004.11165.pdf. Instead of the 4 objectives they
           propose, for now we use 3. The 4th concerns how well the counterfactual fits the
           training data, but since we only choose from the training data, this score would
           alwas be the same. Since our features are highly correlated, their NSGA-II approach
           to perturbing and sampling instances would not be suitable as it is.

           :param data_point: a pandas Series object for the instance we want to explain.
        """
        data_point_X = data_point[self.dataset.feature_names]

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        best_pred, best_prob, cf_pred, cf_prob = self.predictor.get_top2_predictions(data_point_X)
        
        # Find profiles classified as cf prediction.
        candidates = self.predictor.get_data_corr_predicted_as(cf_pred).index

        # Create dataframe to store scores.
        columns = ['target_score', 'distance_score', 'features_changed']
        candidate_scores = pd.DataFrame(columns = columns)

        # Calculate all scores for each candidate data point.
        for candidate in candidates:
            candidate_X = self.dataset.train_data[self.dataset.feature_names].loc[candidate]
            candidate_pred = self.predictor.get_prediction(candidate_X)

            target_score = self.calculate_target_score(candidate_pred, cf_pred)
            distance_score = self.calculate_distance_score(candidate_X, data_point_X)
            features_changed = self.calculate_features_changed(candidate_X, data_point_X)
            
            # Put together into a series and append to dataframe as as row.
            new_row = pd.Series({'target_score':target_score, 
                                 'distance_score':distance_score, 
                                 'features_changed':features_changed},
                                  name=candidate)
            candidate_scores = candidate_scores.append(new_row)
        
        # Create counterfactuals from non-dominated profiles.
        non_dominated_profiles = self.get_non_dominated(candidate_scores)
        for profile in non_dominated_profiles:
            # We do not want counterfactuals with 10 or more feature changes.
            if candidate_scores['features_changed'].loc[profile] < 10:
                # Create counterfactual instance.
                self.counterfactuals = Counterfactual(data_point_X, best_pred, best_prob,
                                                      cf_pred, cf_prob, 
                                                      self.dataset.train_data[self.dataset.feature_names].loc[profile], 
                                                      candidate_scores.loc[profile],
                                                      self.predictor.model_name)
                
        
    def calculate_target_score(self, candidate_pred, target_pred):
        """Score between counterfactual (cf) candidate pred score and cf target.
           Note that the candidate score can be a regression score, so that 
           e.g. 4.2 is closer to 4 than 4.4., even though their classifications are the same.
           
        :param candidate_pred: float representing the counterfactual (cf) candidate's prediction.
        :param target_pred: float representing the target prediction.
        """

        return abs(target_pred - candidate_pred)

    def calculate_MAD_distance_score(self, candidate, data_point):
        """Distance score between the original data point and the counterfactual (cf) candidate. 
           We are using the MAD-scaled score: eq 4 in Wachter et al. https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf
           WARNING: only useful for normally distributed features!

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.
        """

        features_mad = self.dataset.get_features_mad()
        score = 0
        for feature in features_mad:
            dist = abs(candidate[feature] - data_point[feature])
            mad_scaled_dist = dist / (features_mad[feature])
            score += mad_scaled_dist
        return score

    def calculate_distance_score(self, candidate, data_point):
        """Distance score between the original data point and the counterfactual (cf) candidate. 
           We are using the range-scaled score: eq 1 in Dandl et al. https://arxiv.org/pdf/2004.11165.pdf

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.
        """

        features_min_max = self.dataset.get_features_min_max()
        score = 0
        for feature in features_min_max.columns:
            dist = abs(candidate[feature] - data_point[feature])
            max_scaled_dist = dist / (features_min_max[feature].loc['max'])
            score += max_scaled_dist
        return score

    def calculate_features_changed(self, candidate, data_point):
        """Count of how many features are changed between the original data point and the 
           counterfactual (cf) candidate. Note that even if the value is only slightly off, 
           this will count towards this score. 

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.
        """
        
        count = 0
        for c, t in zip(candidate, data_point):
            if c != t:
                count += 1
        return count
    
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

class Counterfactual:

    def __init__(self, data_point, pred, prob, 
                 target_pred, target_prob, counterfactual, scores, model_name):
        """Init method

        :param data_point: pandas series of the data point we are explaining with this counterfactual.
        :param pred: float for the data point prediction.
        :param prob: float for the probability of the data point prediction.
        :param target_pred: float for the target prediction.
        :param target_prob: float for the probability of the target prediction.
        :param counterfactual: pandas series of the counterfactual data point.
        :param scores: dataframe of the 3 scores for this counterfactual.
        :param model_name: string representing the model.

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

        if type(target_pred) is float or isinstance(target_pred, np.float64):
            self.target_pred = target_pred
        else:
            raise ValueError("should provide target prediction as a float")

        if type(target_prob) is float or isinstance(target_prob, np.float64):
            self.target_prob = target_prob
        else:
            raise ValueError("should provide target prediction probability as a float")

        if isinstance(counterfactual, pd.Series):
            self.counterfactual = counterfactual
        else: 
            raise ValueError("should provide counterfactual data point in a pandas series")

        if isinstance(scores, pd.Series):
            self.scores = scores
        else: 
            raise ValueError("should provide scores in a pandas series")

        if type(model_name) is str:
            self.model_name = model_name
        else:
            raise ValueError("should provide model name as a string")

        self.changes = self.calculate_changes()
        self.plot_own_axis()

    def calculate_changes(self):
        """Calculate the pairwise changes between data point and counterfactual."""

        # Rename to match data point (so it doesn't come up as a difference).
        counterfactual = self.counterfactual
        counterfactual.rename({counterfactual.name:self.data_point.name}, inplace=True)

        # Only keep features that are different.
        compare = self.data_point.compare(counterfactual)
        
        # Add column to show how the data point would need to change to become the counterfactual.
        diff_column = (compare["other"] - compare["self"])
        compare["difference"] = diff_column
        
        return compare

    def print_table(self):
        """Print the Counterfactual as a table."""

        row = {}
        for feature in self.counterfactual.index:
            # Only report counterfactual value of a feature if it differs.
            if self.data_point[feature] != self.counterfactual[feature]:
                row[feature] = self.counterfactual[feature]
            else:
                row[feature] = ''
        
        counterfactual_series = pd.Series(row)
        # Possibly change this name to be 'counterfactual option'.
        counterfactual_series.name = self.counterfactual.name
        table = pd.DataFrame([self.data_point, counterfactual_series])

        print(table.T)

    def get_title(self):
        """Define a title string based on the Counterfactual."""
        
        title = ('The profile ' + self.data_point.name + 
                ' was predicted by model ' + self.model_name +
                ' to have {}'.format(int(self.pred)) +
                ' contributors with a probability of {:.2f}'.format(self.prob) +
                '. The next best prediction for this profile is {}'.format(int(self.target_pred)) +
                ' contributors, with a probability of {:.2f}'.format(self.target_prob) +
                '.\n Profile ' + self.data_point.name +  
                ' would have been predicted to have {}'.format(int(self.target_pred)) + 
                ' contributors, if it had the feature values shown below' +
                ' (while keeping the features not in this plot unchanged).')
        return title

    def get_figure_handles(self):
        input_color = mpatches.Patch(label='Profile feature value', color='tab:gray')
        increase_by = mpatches.Patch(label='Increase by', facecolor='w', 
                                     hatch='.', edgecolor='tab:green')
        decrease_by = mpatches.Patch(label='Decrease by', facecolor='tab:gray', 
                                     hatch ='\\', edgecolor ='tab:red')
        return [input_color, increase_by, decrease_by]

    def plot_own_axis(self, scale_y=False):
        """Create a figure where each feature value change from data point to counterfactual
           has their own axis.
        """

        # Only include changed feature values.        
        changed_features = self.changes.index
        
        # Set up figure.
        fig, ax = plt.subplots(nrows=1, ncols=len(changed_features), figsize=(15, 5))
        fig.suptitle(self.get_title())
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(len(changed_features)):
            feature_row = self.changes.iloc[i]
            feature_name = feature_row.name
            dp_val = feature_row.self # data point value
            change = feature_row.difference

            # Input value needs to be increased to match counterfactual.
            if change > 0: 
                bottom = ax[i].bar(feature_name, dp_val, color='tab:gray', edgecolor='tab:gray')
                top = ax[i].bar(feature_name, change, bottom=dp_val, color='w', hatch='.', edgecolor='tab:green')
                ax[i].text(top[0].get_x() + top[0].get_width()/2., 
                           1.04*(top[0].get_height() + bottom[0].get_height()), 
                           '+{:.4g}'.format(change), ha='center', va='top')
            # Input value needs to be decreased to match counterfactual.
            else: 
                top = ax[i].bar(feature_name, dp_val, color='tab:gray', edgecolor='tab:gray')
                ax[i].bar(feature_name, abs(change), bottom=dp_val+change, color='tab:gray', hatch ='\\', edgecolor ='tab:red')
                ax[i].text(top[0].get_x() + top[0].get_width()/2., 
                           1.04*(top[0].get_height()), '{:.4g}'.format(change),
                           ha='center', va='top')
            # TODO: allow this option
            # y axis to represent the min and max values of a feature
            # ax[i].set_ylim(self.min_max[feature_name])
        
        # Adding legend to figure and saving it.
        fig.legend(handles=self.get_figure_handles(), ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout()
        plt.savefig("counterfactual_v1.png", facecolor=fig.get_facecolor(), bbox_inches='tight')

    def plot_one_axis(self, scale_y=False):
        """Create a figure where each feature value change from data point to counterfactual
           is in one figure.
        """
        raise NotImplementedError()
        #TODO: Implement! Would need min/max values of features.
    
    ############################# one-axis plot ################################
    # def plot_counterfactual_v2(self, counterfactual, current_X, current_y, prob_1, target, prob_2):
        
    #     fig, ax = plt.subplots(figsize=(35, 10))
    #     fig.suptitle(self.get_title())
    #     fig.patch.set_facecolor('#E0E0E0')

    #     # Create a list of ones to represent the baseline
    #     num_features = len(current_X.index)

    #     # Create a list of medians
    #     current_X_scaled = []
    #     for feature_name in current_X.index:
    #         med = self.min_max[feature_name].values[2]
    #         cur = current_X.loc[feature_name]
    #         var_from_med = cur - med

    #         if var_from_med == 0:
    #             current_X_scaled.append(1)
    #         elif med == 0:
    #             current_X_scaled.append(var_from_med)
    #         else:
    #             current_X_scaled.append(1 + (var_from_med / med))
        
    #     # Plot the original
    #     original = ax.bar(current_X.index, current_X_scaled, color='tab:gray', 
    #                       edgecolor='tab:gray')
    #     #plt.yscale('log')

    #     # Get the container for the barchart
    #     container = ax.containers[0]

    #     # Loop over the individual bars
    #     for i, curr_bar in enumerate(container.get_children()):
    #         feature_name = current_X.index[i]
    #         # Check if feature was changed
    #         if feature_name in counterfactual.index:
    #             # Check if positive or negative diff
    #             diff = counterfactual.loc[feature_name].difference
    #             curr_val = counterfactual.loc[feature_name].self
    #             # Input value needs to be increased to match counterfactual.
    #             # if diff > 0:
    #             #     top = ax.bar(feature_name, diff, bottom=curr_val, color='w',
    #             #             hatch='.', edgecolor='tab:green')
    #             #     ax.text(top[0].get_x() + top[0].get_width()/2., 
    #             #        1.2*(top[0].get_height() + curr_bar.get_height()), 
    #             #        '+{:.4g}'.format(diff), ha='center', va='top')
    #             # # Input value needs to be decreased to match counterfactual.
    #             # else:
    #             #     ax.bar(feature_name, abs(diff), bottom=curr_val+diff, 
    #             #         color='tab:gray', hatch ='\\', edgecolor ='tab:red')
    #             #     ax.text(curr_bar.get_x() + curr_bar.get_width()/2., 
    #             #         1.2*(curr_bar.get_height()), '{:.4g}'.format(diff),
    #             #         ha='center', va='top')


    #             # if diff > 0:
    #             #     top = ax.bar(feature_name, diff/curr_val, bottom=1, color='w',
    #             #             hatch='.', edgecolor='tab:green')
    #             #     ax.text(top[0].get_x() + top[0].get_width()/2., 
    #             #        1.03*(top[0].get_height() + curr_bar.get_height()), 
    #             #        '+{:.4g}'.format(diff), ha='center', va='top')
    #             # # Input value needs to be decreased to match counterfactual.
    #             # else:
    #             #     ax.bar(feature_name, abs(diff)/curr_val, bottom=1-abs(diff)/curr_val, 
    #             #         color='tab:gray', hatch ='\\', edgecolor ='tab:red')
    #             #     ax.text(curr_bar.get_x() + curr_bar.get_width()/2., 
    #             #         1.03*(curr_bar.get_height()), '{:.4g}'.format(diff),
    #             #         ha='center', va='top')


    #             if diff > 0:
    #                 top = ax.bar(feature_name, diff/curr_val, bottom=curr_bar.get_height(), color='w',
    #                         hatch='.', edgecolor='tab:green')
    #                 ax.text(top[0].get_x() + top[0].get_width()/2., 
    #                    1.03*(top[0].get_height() + curr_bar.get_height()), 
    #                    '+{:.4g}'.format(diff), ha='center', va='top')
    #             # Input value needs to be decreased to match counterfactual.
    #             else:
    #                 ax.bar(feature_name, abs(diff)/curr_val, bottom=curr_bar.get_height()-abs(diff)/curr_val, 
    #                     color='tab:gray', hatch ='\\', edgecolor ='tab:red')
    #                 ax.text(curr_bar.get_x() + curr_bar.get_width()/2., 
    #                     1.03*(curr_bar.get_height()), '{:.4g}'.format(diff),
    #                     ha='center', va='top')


    #     input_color = mpatches.Patch(label='Profile feature value', color='tab:gray')
    #     increase_by = mpatches.Patch(label='Increase by', facecolor='w', hatch='.', edgecolor='tab:green')
    #     decrease_by = mpatches.Patch(label='Decrease by', facecolor='tab:gray', hatch ='\\', edgecolor ='tab:red')
    #     fig.legend(handles=[input_color, increase_by, decrease_by], ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.05))
    #     fig.tight_layout()
    #     plt.savefig("counterfactual_v2.png", facecolor=fig.get_facecolor(), bbox_inches='tight')

