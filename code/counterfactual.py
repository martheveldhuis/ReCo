import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

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
        data_point_scaled = pd.Series(self.dataset.scaler.transform(data_point_X.to_numpy().reshape(1, -1)).ravel())
        data_point_scaled.name = data_point_X.name
        data_point_scaled.index = data_point_X.index

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
                print('profile '+profile+ 'with {}'.format(candidate_scores['features_changed'].loc[profile]) +
                ' features changed and {}'.format(candidate_scores['distance_score'].loc[profile]) + 
                ' distance score')
                self.counterfactuals = Counterfactual(data_point_X, data_point_scaled, best_pred, best_prob,
                                                      cf_pred, cf_prob, 
                                                      self.dataset.train_data[self.dataset.feature_names].loc[profile], 
                                                      self.dataset.scaled_train_data[self.dataset.feature_names].loc[profile],
                                                      self.predictor.model_name)
                
    def generate_local_avg_train_counterfactual(self, data_point, n, shap_values):
        """Generate a counterfactual by taking the average of n closest local training points with 
           the target prediction.

           :param data_point: a pandas Series object for the instance we want to explain.
           :param n: the number of training points to calculate the average from.
        """
        feature_names = self.dataset.feature_names
        data_point_X = data_point[feature_names]
        data_point_scaled = pd.Series(self.dataset.scaler.transform(data_point_X.to_numpy().reshape(1, -1)).ravel())
        data_point_scaled.name = data_point_X.name
        data_point_scaled.index = data_point_X.index

        # Get the data point's top and second-best, counterfactual (cf), prediction.
        best_pred, best_prob, cf_pred, cf_prob = self.predictor.get_top2_predictions(data_point_X)
        
        # Find profiles classified as cf prediction.
        candidates = self.predictor.get_data_corr_predicted_as(cf_pred).index
        candidates_X = self.dataset.train_data[feature_names].loc[candidates]
        candidate_predictions = self.predictor.get_prediction(candidates_X)

        # Create dataframe to store scores.
        columns = ['target_score', 'distance_score', 'features_changed']
        candidate_scores = pd.DataFrame(columns = columns)

        # Calculate all scores for each candidate data point.
        i=0
        for candidate in candidates:
            candidate_X = candidates_X.loc[candidate]
            candidate_pred = candidate_predictions[i]

            target_score = self.calculate_target_score(candidate_pred, best_pred)
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
        candidate_scores['sum'] = candidate_scores.sum(axis=1)
        candidate_scores.sort_values(by='target_score', inplace=True)
        print(candidate_scores.iloc[0])
        print(candidate_scores.iloc[1])

        # Get top counterfactual from non-dominated profiles.
        non_dominated_profiles = self.get_non_dominated(candidate_scores)
        non_dom_candidates = candidate_scores.loc[non_dominated_profiles]
        cf = non_dom_candidates.iloc[0].copy()

        # Calculate the average feature values of the n top candidates.
        top_n_names = candidate_scores.head(n).index
        top_n = self.dataset.train_data.loc[top_n_names]
        avg_instance = top_n.median(axis=0)[feature_names]
        avg_instance.name = 'avg{}'.format(n)
        avg_instance_scaled = pd.Series(self.dataset.scaler.transform(avg_instance.to_numpy().reshape(1, -1)).ravel())
        avg_instance_scaled.index = feature_names
        avg_instance_scaled.name = 'avg{}'.format(n)

        # Get the CF sample and tune it.
        original_cf = self.dataset.train_data[self.dataset.feature_names].loc[cf.name].copy()

        # i = 0
        # for d_v, cf_v in zip(data_point_X, original_cf):
        #     feature_name = feature_names[i]
        #     if d_v != cf_v:
        #         direction_cf = d_v - cf_v
        #         direction_avg = d_v - avg_instance[feature_name]
        #         if direction_cf * direction_avg < 0.0: # Check that we are not going against avg.
        #             original_cf[feature_name] = d_v
        #     i+=1

        # Check the prediction
        print('new prediction: ')
        print(self.predictor.get_prediction(original_cf))

        scaled_cf = pd.Series(self.dataset.scaler.transform(original_cf.to_numpy().reshape(1, -1)).ravel())
        scaled_cf.index = feature_names
        
        # Create counterfactual
        # self.counterfactuals = Counterfactual(data_point_X, data_point_scaled, best_pred, best_prob,
        #                                       cf_pred, cf_prob, avg_instance, avg_instance_scaled,
        #                                       self.predictor.model_name)
        self.counterfactuals = Counterfactual(data_point_X, data_point_scaled, best_pred, best_prob,
                                                cf_pred, cf_prob, original_cf, scaled_cf, 
                                                self.predictor.model_name, shap_values)
        
        return avg_instance


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

        num_features_changed = 0
        features_min_max = self.dataset.get_features_min_max()
        score = 0        
        for feature in features_min_max.columns:
            dist = abs(candidate[feature] - data_point[feature])
            max_scaled_dist = dist / (features_min_max[feature].loc['max'])
            if max_scaled_dist > 0.05: # Allow for 5% tolerance
                score += max_scaled_dist
                num_features_changed += 1
        score = score / len(features_min_max.columns)
        changed = num_features_changed / len(features_min_max.columns)
        return score, changed

    def calculate_features_changed(self, candidate, data_point):
        """Count of how many features are changed between the original data point and the 
           counterfactual (cf) candidate. Note that even if the value is only slightly off, 
           this will count towards this score. 

        :param candidate: the counterfactual(cf) candidate's feature values.
        :param data_point: the data point's feature values.
        """
        
        num_features = len(data_point.index)
        count = 0
        for c, t in zip(candidate, data_point):
            if c != t:
                count += 1
        return count/num_features
    
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

    def __init__(self, data_point, data_point_scaled, pred, prob, 
                 target_pred, target_prob, counterfactual, counterfactual_scaled, 
                 model_name, shap_values):
        """Init method

        :param data_point: pandas series of the data point we are explaining with this counterfactual.
        :param data_point_scaled: pandas series of the scaled data point we are explaining with this counterfactual.
        :param pred: float for the data point prediction.
        :param prob: float for the probability of the data point prediction.
        :param target_pred: float for the target prediction.
        :param target_prob: float for the probability of the target prediction.
        :param counterfactual: pandas series of the counterfactual data point.
        :param counterfactual_scaled: pandas series of the scaled counterfactual data point.
        :param model_name: string representing the model.

        """
        if isinstance(data_point, pd.Series):
            self.data_point = data_point
        else: 
            raise ValueError("should provide data point in a pandas series")

        if isinstance(data_point_scaled, pd.Series):
            self.data_point_scaled = data_point_scaled
        else: 
            raise ValueError("should provide scaled data point in a pandas series")

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

        if isinstance(counterfactual_scaled, pd.Series):
            self.counterfactual_scaled = counterfactual_scaled
        else: 
            raise ValueError("should provide scaled counterfactual data point in a pandas series")

        if type(model_name) is str:
            self.model_name = model_name
        else:
            raise ValueError("should provide model name as a string")

        self.changes = self.calculate_changes()
        #self.plot_own_axis()
        self.plot_one_axis(shap_values)

    def calculate_changes(self):
        """Calculate the pairwise changes between data point and counterfactual."""

        # Rename to match data point (so it doesn't come up as a difference).
        counterfactual = self.counterfactual.copy()
        data_point = self.data_point.copy()
        counterfactual.name = ''
        data_point.name = ''

        # Only keep features that are different.
        compare = data_point.compare(counterfactual)
        
        # Add column to show how the data point would need to change to become the counterfactual.
        diff_column = (compare["other"] - compare["self"])
        compare["difference"] = diff_column
        
        return compare

    def calculate_scaled_changes(self):
        """Calculate the pairwise changes between data point and counterfactual."""

        # Rename to match data point (so it doesn't come up as a difference).
        counterfactual = self.counterfactual_scaled.copy()
        data_point = self.data_point_scaled.copy()
        counterfactual.name = ''
        data_point.name = ''

        # Only keep features that are different.
        compare = data_point.compare(counterfactual)
        
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

    def get_title(self, classification=False):
        """Define a title string based on the Counterfactual."""
        
        # For classification, we can add the probabilities of the first and second predictions.
        probability_string = ''
        if(classification):
            probability_string = (' with a probability of {:.2f}'.format(self.prob) +
                            '. The next best prediction is {}'.format(int(self.target_pred)) +
                            ' contributors, with a probability of {:.2f}'.format(self.target_prob))

        title = ('The profile ' + self.data_point.name + 
                ' was predicted by model ' + self.model_name +
                ' to have {}'.format(self.pred) + 
                ' contributors' + probability_string +
                '.\nProfile ' + self.data_point.name +  
                ' would have been predicted to have {}'.format(round(self.target_pred)) + 
                ' contributors, if all following feature values were different.')
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
        fig.suptitle(self.get_title(), x=0.13, ha='left')
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
        plt.savefig(r'results\counterfactual_v1.png', facecolor=fig.get_facecolor(), bbox_inches='tight')

    def plot_one_axis(self, shap_values):
        """Create a figure where each feature value change from data point to counterfactual
           is in one figure.
        """

        # Grab the values we need.
        dp = self.data_point
        dp_scaled = self.data_point_scaled
        cf = self.counterfactual
        cf_scaled = self.counterfactual_scaled
        
        # Set up figure.
        fig, ax = plt.subplots(figsize=(18,9), constrained_layout=False) # this might need to be changed when dealing with more/less features.
        fig.suptitle(self.get_title(), x=0.13, ha='left')
        fig.patch.set_facecolor('#E0E0E0')
        
        # Set up colors.
        # colormap = plt.get_cmap('RdBu') # looks too much like our arrow colors
        # colormap = plt.get_cmap('BrBG')
        colormap = plt.get_cmap('coolwarm_r')
        offset = mcolors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8)
        colors = offset(shap_values)    

        

        # Plot the bars for the scaled data point
        dp_bars = ax.barh(dp.index, dp_scaled, color=colormap(colors), alpha=1, edgecolor='#E0E0E0')
        ax.set_xticklabels([]) # remove x-values
        ax.set_xlim([0,1.06]) # ensure the scale is always the same, and fits the text (+0.06)
        plt.gca().invert_yaxis() # put first feature at the top
        labels = ax.get_yticklabels()

        # Create a color bar legend.
        mappable = plt.cm.ScalarMappable(norm=offset, cmap=colormap)
        mappable.set_array([])
        colorbar = plt.colorbar(mappable, shrink = 0.8, label = 'Influence of feature values on this prediction')
        colorbar.set_ticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
        cb_labels = ['\N{MINUS SIGN}0.8','\N{MINUS SIGN}0.6',
                     '\N{MINUS SIGN}0.4','\N{MINUS SIGN}0.2',
                     '0.0', '+0.2', '+0.4', '+0.6', '+0.8']
        colorbar.set_ticklabels(cb_labels)

        # Get the scaled changes.
        changes = self.calculate_scaled_changes()

        for i, dp_bar in enumerate(ax.containers[0].get_children()):
            dp_feature_name = dp.index[i]
            dp_feature_val = dp[i]
            dp_scaled_val = dp_scaled[i]
            cf_feature_val = cf[i]
            cf_scaled_val = cf_scaled[i]
            
            bar_y = dp_bar.get_y()
            bar_height = dp_bar.get_height()
            bar_color = dp_bar.get_facecolor()

            # Add changes to barchart.
            if dp_feature_name in changes.index:
                diff = changes.loc[dp_feature_name].difference
                # Move label over if either value is too close to 0.
                if cf_scaled_val < 0.04 or dp_scaled_val < 0.04:
                    labels[i].set_x(labels[i].get_position()[0] - 0.06) 
                # Input value needs to be increased to match counterfactual.
                if diff > 0:
                    # Stack on top
                    ax.barh(dp_feature_name, diff, left=dp_scaled_val,
                            color='w', alpha=1, edgecolor='#E0E0E0')
                    # Put data point value text inside bar.
                    ax.text(dp_bar.get_width()-0.01, bar_y+bar_height/2.,
                            '{:.4g}'.format(dp_feature_val), ha='right', va='center')
                    # Put cf data point value text outside bar.
                    ax.text(dp_bar.get_width()+diff+0.01, bar_y+bar_height/2.,
                            '{:.4g}'.format(cf_feature_val), ha='left', va='center')
                    # Add arrow
                    head_length = 0.01 if diff >= 0.01 else 0.75 * diff
                    ax.arrow(dp_scaled_val, bar_y+bar_height/2., diff, 0, width=0.1, color='tab:blue',
                             length_includes_head=True, head_width=0.4, head_length=head_length)
                # Input value needs to be decreased to match counterfactual.
                else:
                    ax.barh(dp_feature_name, abs(diff), left=dp_scaled_val+diff,
                            color=bar_color, alpha=1, edgecolor='w')
                    # Put data point value text outside bar.
                    ax.text(dp_bar.get_width()+0.01, bar_y+bar_height/2.,
                            '{:.4g}'.format(dp_feature_val), ha='left', va='center')
                    # Put cf data point value text inside bar.
                    ax.text(dp_bar.get_width()+diff-0.01, bar_y+bar_height/2.,
                            '{:.4g}'.format(cf_feature_val), ha='right', va='center')
                    # Add arrow
                    head_length = 0.01 if abs(diff) >= 0.01 else 0.75 * abs(diff)
                    ax.arrow(dp_scaled_val, bar_y+bar_height/2., diff, 0, width=0.1, color='tab:orange',
                             length_includes_head=True, head_width=0.4, head_length=head_length)
            else:
                # Plot data point value.
                ax.text(dp_bar.get_width()+0.01, bar_y+bar_height/2.,
                        '{:.4g}'.format(dp_feature_val), ha='left', va='center')
        
        # Save figure.
        fig.tight_layout()
        plt.savefig(r'results\cf_' + dp.name + '_' + cf.name + '.png', 
                    facecolor=fig.get_facecolor(), bbox_inches='tight')

