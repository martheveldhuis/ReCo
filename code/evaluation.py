import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import dice_ml
from dice_ml.utils import helpers # helper functions

from data import Dataset
from predictor import Predictor
from counterfactual import CounterfactualGenerator
from shap_values import ShapGenerator

class Evaluator:
    """A class for evaluating counterfactuals."""

    def __init__(self, dataset, predictor, cf_generator, shap_generator):
        """Init method

        :param dataset: Dataset instance containing all data information.
        :param predictor: Predictor instance wrapping all predictor functionality.
        :param cf_generator: CounterfactualGenerator instance wrapping all cf functionality.
        :param shap_generator: ShapGenerator instance wrapping all SHAP functionality.
        """

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError('should provide data as Dataset instance')

        if isinstance(predictor, Predictor):
            self.predictor = predictor
        else:
            raise ValueError('should provide predictor as Predictor instance')

        if isinstance(cf_generator, CounterfactualGenerator):
            self.cf_generator = cf_generator
        else:
            raise ValueError('should provide cf generator as CounterfactualGenerator instance')

        if isinstance(shap_generator, ShapGenerator):
            self.shap_generator = shap_generator
        else:
            raise ValueError('should provide shap generator as ShapGenerator instance')

    def get_whatif_counterfactual(self, dp, dp_scaled, cf_target, i):

        return self.cf_generator.generate_whatif_counterfactual(dp, dp_scaled, cf_target)

    def get_dice_random_counterfactual(self, dp, dp_scaled, cf_target, i):

        if cf_target == 1.0:
            range = [1.0, 1.4]
        elif cf_target == 5.0:
            range = [4.5, 5.0]
        else:
            range = [cf_target - 0.5, cf_target + 0.4]

        query_instances = self.dataset.test_data[i:i+1][self.dataset.feature_names]

        dice_exp_random = self.exp.generate_counterfactuals(query_instances, total_CFs=1, desired_range=range)
        cf_profile = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.feature_names]
        cf_pred = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.outcome_name]
        cf_scaled = self.dataset.scale_data_point(cf_profile)
        changes = self.cf_generator.calculate_scaled_changes(cf_scaled, dp_scaled)

        return cf_profile, cf_scaled, cf_pred, changes

    def get_dice_genetic_counterfactual(self, dp, dp_scaled, cf_target, i):

        if cf_target == 1.0:
            range = [1.0, 1.4]
        elif cf_target == 5.0:
            range = [4.5, 5.0]
        else:
            range = [cf_target - 0.5, cf_target + 0.4]

        query_instances = self.dataset.test_data[i:i+1][self.dataset.feature_names]

        dice_exp_genetic= self.exp.generate_counterfactuals(query_instances, total_CFs=1, 
                                                            desired_range=range, 
                                                            permitted_range=self.permitted_range)
        cf_profile = dice_exp_genetic.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.feature_names]
        cf_pred = dice_exp_genetic.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.outcome_name]
        cf_scaled = self.dataset.scale_data_point(cf_profile)
        changes = self.cf_generator.calculate_scaled_changes(cf_scaled, dp_scaled)

        return cf_profile, cf_scaled, cf_pred, changes

    def get_reco_unfiltered_counterfactual(self, dp, dp_scaled, cf_target, i):

        return self.cf_generator.generate_weighted_counterfactual(dp, dp_scaled, cf_target)

    def get_reco_counterfactual(self, dp, dp_scaled, cf_target, i):

        cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_weighted_counterfactual(dp, dp_scaled, cf_target)
        dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
        dp_shap = self.shap_generator.get_shap_values(dp)
        cf_shap = self.shap_generator.get_shap_values(cf_profile)
        changes = self.cf_generator.add_shap_tolerance(dp, dp_shap, dp_pred, cf_profile, cf_shap, cf_pred, changes)

        cf = dp.copy()
        for feature, row in changes.iterrows():
            cf[feature] = cf_profile[feature]
        new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])
        
        return cf, cf_scaled, new_pred, changes


    def evaluate(self, method):
        profiles = []
        features_changed = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []
        
        # Methods that we can evaluate.
        method_dict =  {
            'dice_random': self.get_dice_random_counterfactual,
            'dice_genetic': self.get_dice_genetic_counterfactual,
            'whatif': self.get_whatif_counterfactual,
            'reco_unfiltered': self.get_reco_unfiltered_counterfactual,
            'reco': self.get_reco_counterfactual
        }
        cf_method = method_dict.get(method, 'invalid method')

        # Some pre-loop dice stuff to set.
        if method == 'dice_random' or method == 'dice_genetic':
            continuous = self.dataset.feature_names

            d = dice_ml.Data(dataframe=self.dataset.train_data.copy(), 
                             continuous_features=continuous, 
                             outcome_name=self.dataset.outcome_name)
            m = dice_ml.Model(model=self.predictor.model, backend='sklearn', model_type='regressor')

            if method == 'dice_random': 
                self.exp = dice_ml.Dice(d, m, method='random')
            else: 
                
                permitted_range = {}
                min_max = self.dataset.get_features_min_max()
                for feature in self.dataset.feature_names:
                    min = min_max[feature].loc['min']
                    max = min_max[feature].loc['max']
                    permitted_range[feature] = [min, max]
                self.permitted_range = permitted_range
                self.exp = dice_ml.Dice(d, m, method='genetic')

        # Loop through all test data points, get its prediction
        i = 0
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            
            # Generate CF with desired method.
            cf_profile, cf_scaled, cf_pred, changes = cf_method(dp, dp_scaled, cf_target, i)
            cf = cf_profile.copy()
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # Calculate the scores
            features_changed.append(self.count_features_changed(changes))
            distance, realism_score = (self.get_cf_scores(changes, dp, dp_pred, cf, cf_pred))
            distance_to_dp.append(distance)
            realism.append(realism_score)
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            # Prints for monitoring progress.
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target) + 
                  ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            print('features changed: {:.2f}'.format(features_changed[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        # Put all scores together and write to file (name is now a placeholder).
        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice_genetic.csv')


    def count_features_changed(self, changes):
        # The number of changed features displayed to the user.
        return changes.shape[0]


    def get_cf_scores(self, changes, dp, dp_pred, cf, cf_pred):
        feat_min_max = self.dataset.get_features_min_max()
        dist_to_dp = 0
        i = 0
        realism_score = 0
        for feature_name in self.dataset.feature_names:
            if feature_name in changes.index:
                
                # Calculate the distance based on min/max values
                dist = abs(cf[feature_name] - dp[feature_name])
                feat_range = feat_min_max[feature_name].loc['max'] - feat_min_max[feature_name].loc['min']
                scaled_dist = dist / feat_range
                dist_to_dp += scaled_dist

                # Get the realism score 
                feature_realism_score = 0
                correlated_features_checked = 0                   
                for corr_feature in self.dataset.get_top_corr_features(feature_name):
                    cf_feature_val_data = self.dataset.train_data[self.dataset.train_data[feature_name] == cf[feature_name]]
                    if corr_feature in changes.index:
                        combo_data = cf_feature_val_data[cf_feature_val_data[corr_feature] == cf[corr_feature]]
                    else:
                        combo_data = cf_feature_val_data[cf_feature_val_data[corr_feature] == dp[corr_feature]]
                    if combo_data.shape[0] > 0:
                       feature_realism_score += 1
                    correlated_features_checked +=1
                    if corr_feature not in changes.index: # make sure at least 1 corr feature was in the original profile
                        realism_score += feature_realism_score/correlated_features_checked  
                        break                          
                
            i+=1

        dist_to_dp /= 19
        realism_score /= changes.shape[0]
        return dist_to_dp, realism_score

    def get_target_missed(self, cf_target, cf_pred):
        # If the target was missed or not (1 if missed; so cf_target not equal to pred)
        return cf_target != cf_pred.round()

    def get_distance_to_td(self, cf, cf_target):
        # Get the closest distance to a training data point
        training_data = self.predictor.get_data_corr_predicted_as(cf_target)[self.dataset.feature_names]
        feat_min_max = self.dataset.get_features_min_max()

        min_dist = np.inf
        for index, training_dp in training_data.iterrows():
            total_dist = 0
            for feature_name in self.dataset.feature_names:
                feat_range = feat_min_max[feature_name].loc['max'] - feat_min_max[feature_name].loc['min']
                dist = abs(cf[feature_name] - training_dp[feature_name])
                scaled_dist = dist / feat_range
                total_dist += scaled_dist
            
            if total_dist < min_dist:
                min_dist = total_dist

        return min_dist/19

    def print_scores(self):

        filenames = [r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice_genetic',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_whatif',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_reco_no_f',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_reco']
        
        for file in filenames:
            eval = pd.read_csv(file + '.csv', index_col=0)
            print('Features changed avg: {}'.format(np.mean(eval['Features changed'])))   
            print('Dist to profile: {}'.format(np.mean(eval['Distance to data point'])))
            print('Dist to training point: {}'.format(np.mean(eval['Distance to training data point'])))
            print('target missed: {}'.format(np.sum(eval['Target missed'])))
            print('realism score: {}'.format(np.mean(eval['Realism score'])))

    def plot_boxplot(self): 
        
        filenames = [r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice_genetic',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_whatif',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_reco_no_f',
                     r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_reco']

        eval_dice = pd.read_csv(filenames[0], index_col=0)
        eval_dice_g = pd.read_csv(filenames[1], index_col=0)
        eval_whatif = pd.read_csv(filenames[2], index_col=0)
        # eval_reco_no_f = pd.read_csv(filenames[3], index_col=0)
        eval_reco = pd.read_csv(filenames[4], index_col=0)
        
        fig, axss = plt.subplots(2, 2, figsize=(11,7))

        box_colors = ['#1170aa', '#5fa2ce', '#a3acb9', '#ffbc79'] # '#c85200',
        titles = ['Number of features changed', 'Distance to the profile to be explained', 
                  'Distance to the nearest training data point', 'Targets missed', 'Realism score']

        i = 1
        for axs in axss:
            for ax in axs:
                if i == 4: # Skipping targets missed column in files
                    i = 5
                
                bplot = ax.boxplot([eval_dice.iloc[:,i], eval_dice_g.iloc[:,i], 
                                    eval_whatif.iloc[:,i], eval_reco.iloc[:,i]], patch_artist=True,
                                    medianprops=dict(color='k', linewidth=2))
                ax.set_title(titles[i-1], pad=15)
                ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
                ax.set_xticklabels(['DiCE random', 'DiCE genetic', 'WhatIf', 'ReCo'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for patch, color in zip(bplot['boxes'], box_colors):
                    patch.set_facecolor(color)
                i += 1

        fig.tight_layout(pad=2)
        fig.savefig(r'evaluation\boxplot_final_eval.png')
