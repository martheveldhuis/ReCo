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
            raise ValueError("should provide data as Dataset instance")

        if isinstance(predictor, Predictor):
            self.predictor = predictor
        else:
            raise ValueError("should provide predictor as Predictor instance")

        if isinstance(cf_generator, CounterfactualGenerator):
            self.cf_generator = cf_generator
        else:
            raise ValueError("should provide cf generator as CounterfactualGenerator instance")

        if isinstance(shap_generator, ShapGenerator):
            self.shap_generator = shap_generator
        else:
            raise ValueError("should provide shap generator as ShapGenerator instance")

    def evaluate_whatif(self):
        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        i = 0
        # For each data point in the test set, scale it, get its prediction, generate its SHAP values and a CF
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)
            cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_whatif_counterfactual(dp, dp_scaled, cf_target)
            cf_shap = self.shap_generator.get_shap_values(cf_profile)

            cf = cf_profile.copy()
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            realism.append(realism_score)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg distance to td: {:.2f}'.format(np.mean(distance_to_td)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_whatif.csv')

    def evaluate_dice(self):
        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        continuous = []

        d = dice_ml.Data(dataframe=self.dataset.train_data.copy(), continuous_features=continuous, outcome_name=self.dataset.outcome_name)
        m = dice_ml.Model(model=self.predictor.model, backend='sklearn', model_type='regressor')
        exp_random = dice_ml.Dice(d, m, method='random')

        

        i = 0
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)

            if cf_target == 1.0:
                range = [1.0, 1.4]
            elif cf_target == 5.0:
                range = [4.5, 5.0]
            else:
                range = [cf_target-0.5, cf_target+0.4]

            query_instances = self.dataset.test_data[i:i+1][self.dataset.feature_names]

            dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=1, desired_range=range)
            cf = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.feature_names]
            cf_pred = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.iloc[0][self.dataset.outcome_name]
            cf_scaled = self.dataset.scale_data_point(cf)
            changes = self.cf_generator.calculate_scaled_changes(cf_scaled, dp_scaled)
            cf_shap = self.shap_generator.get_shap_values(cf)

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            realism.append(realism_score)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(cf_pred.round()) + ' unrounded: {:.2f}'.format(cf_pred))
            target_missed.append(self.get_target_missed(cf_target, cf_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_dice.csv')

    def evaluate_pareto_filtered(self):

        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        i = 0
        # For each data point in the test set, scale it, get its prediction, generate its SHAP values and a CF
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)
            cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_pareto_counterfactual(dp, dp_scaled, cf_target)
            cf_shap = self.shap_generator.get_shap_values(cf_profile)
            changes = self.cf_generator.add_shap_tolerance(dp, dp_shap, dp_pred, cf_profile, cf_shap, cf_pred, changes)

            cf = dp.copy()
            for feature, row in changes.iterrows():
                cf[feature] = cf_profile[feature]
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            realism.append(realism_score)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg distance to td: {:.2f}'.format(np.mean(distance_to_td)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_par_f.csv')


    def evaluate_pareto(self):
        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        i = 0
        # For each data point in the test set, scale it, get its prediction, generate its SHAP values and a CF
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)
            cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_pareto_counterfactual(dp, dp_scaled, cf_target)
            cf_shap = self.shap_generator.get_shap_values(cf_profile)

            cf = cf_profile.copy()
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            realism.append(realism_score)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg distance to td: {:.2f}'.format(np.mean(distance_to_td)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_par.csv')

    def evaluate_weight_filtered(self):

        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        i = 0
        # For each data point in the test set, scale it, get its prediction, generate its SHAP values and a CF
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)
            cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_weighted_counterfactual(dp, dp_scaled, cf_target)
            cf_shap = self.shap_generator.get_shap_values(cf_profile)
            changes = self.cf_generator.add_shap_tolerance(dp, dp_shap, dp_pred, cf_profile, cf_shap, cf_pred, changes)

            cf = dp.copy()
            for feature, row in changes.iterrows():
                cf[feature] = cf_profile[feature]
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            realism.append(realism_score)
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_weight_f.csv')

    def evaluate_weight(self):
        profiles = []
        features_changed = []
        perc_features_changed_bad_shap = []
        distance_to_dp = []
        target_missed = []
        distance_to_td = []
        realism = []

        i = 0
        # For each data point in the test set, scale it, get its prediction, generate its SHAP values and a CF
        for index, dp in self.dataset.test_data.iterrows():
            print(i)
            dp_scaled = self.dataset.scale_data_point(dp)
            dp_pred = self.predictor.get_prediction(dp[self.dataset.feature_names])
            cf_target = (self.predictor.get_secondary_prediction(dp[self.dataset.feature_names]))
            dp_shap = self.shap_generator.get_shap_values(dp)
            cf_profile, cf_scaled, cf_pred, changes = self.cf_generator.generate_weighted_counterfactual(dp, dp_scaled, cf_target)
            cf_shap = self.shap_generator.get_shap_values(cf_profile)

            cf = dp.copy()
            for feature, row in changes.iterrows():
                cf[feature] = cf_profile[feature]
            new_pred = self.predictor.get_prediction(cf[self.dataset.feature_names])

            # calculate the scores
            features_changed.append(self.count_features_changed(changes))
            bad_shap, distance, realism_score = (self.get_cf_scores(changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred))
            realism.append(realism_score)
            perc_features_changed_bad_shap.append(bad_shap)
            distance_to_dp.append(distance)
            print('original: {:.2f}'.format(dp_pred) + 'target: {}'.format(cf_target)+ ' new pred: {}'.format(new_pred.round()) + ' unrounded: {:.2f}'.format(new_pred))
            target_missed.append(self.get_target_missed(cf_target, new_pred))
            distance_to_td.append(self.get_distance_to_td(cf, cf_target))
            profiles.append(index)

            print('features changed: {:.2f}'.format(features_changed[i]))
            print('perc features changed with bad shap vals: {:.2f}'.format(perc_features_changed_bad_shap[i]))
            print('distance to dp: {:.2f}'.format(distance_to_dp[i]))
            print('target missed: {}'.format(np.sum(target_missed[i])))
            print('distance to td: {:.2f}'.format(distance_to_td[i]))
            print('realism score: {:.2f}'.format(realism[i]))

            i+=1

        print('avg features changed: {:.2f}'.format(np.mean(features_changed)))
        print('avg perc features changed with bad shap vals: {:.2f}'.format(np.mean(perc_features_changed_bad_shap)))
        print('avg distance to dp: {:.2f}'.format(np.mean(distance_to_dp)))
        print('target missed: {}'.format(np.sum(target_missed)))
        print('avg distance to td: {:.2f}'.format(np.mean(distance_to_td)))
        print('avg realism score: {:.2f}'.format(np.mean(realism)))
        

        evaluation = pd.DataFrame({'Profile': profiles,
                                   'Features changed':features_changed,
                                   'Perc. features changed with bad shap vals': perc_features_changed_bad_shap,
                                   'Distance to data point':distance_to_dp,
                                   'Distance to training data point':distance_to_td,
                                   'Target missed': target_missed,
                                   'Realism score': realism})
        evaluation.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\eval_weight.csv')

    def count_features_changed(self, changes):
        # The number of changed features displayed to the user.
        return changes.shape[0]


    def get_cf_scores(self, changes, dp, dp_shap, dp_pred, cf, cf_shap, cf_pred):
        features_min_max = self.dataset.get_features_min_max()
        dist_to_dp = 0
        i = 0
        bad_shap_counter = 0
        realism_score = 0
        for feature_name in self.dataset.feature_names:
            if feature_name in changes.index:
                # Get the SHAP values
                dp_shap_feature = dp_shap[i]
                cf_shap_feature = cf_shap[i]
                shap_change = cf_shap_feature - dp_shap_feature

                # Increase counter if suggested changes work against the prediction direction
                if ((cf_pred > dp_pred and shap_change <= 0.0) or 
                    (cf_pred < dp_pred and shap_change >= 0.0)): 
                    bad_shap_counter+=1

                # Calculate the distance based on min/max values
                dist = abs(cf[feature_name] - dp[feature_name])
                scaled_dist = dist / (features_min_max[feature_name].loc['max'])
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
        percentage_bad_changes = (bad_shap_counter/changes.shape[0]) * 100
        realism_score /= changes.shape[0]
        return percentage_bad_changes, dist_to_dp, realism_score

    def get_target_missed(self, cf_target, cf_pred):
        # If the target was missed or not (1 if missed; so cf_target not equal to pred)
        return cf_target != cf_pred.round()

    
    def get_distance_to_td(self, cf, cf_target):
        # Get the closest distance to a training data point
        training_data = self.predictor.get_data_corr_predicted_as(cf_target)[self.dataset.feature_names]
        features_min_max = self.dataset.get_features_min_max()

        min_dist = np.inf
        for index, training_dp in training_data.iterrows():
            total_dist = 0
            for feature_name in self.dataset.feature_names:
                dist = abs(cf[feature_name] - training_dp[feature_name])
                scaled_dist = dist / (features_min_max[feature_name].loc['max'])
                total_dist += scaled_dist
            
            if total_dist < min_dist:
                min_dist = total_dist

        return min_dist/19


    def plot_num_features(self, filename):

        eval = pd.read_csv(filename + '.csv', index_col=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.patch.set_facecolor('#E0E0E0')

        ax.hist(eval['Features changed'], bins=100)
        print('Features changed avg: {}'.format(np.mean(eval['Features changed'])))
        plt.title('Features changed')
        plt.savefig(filename + '_fn.png', facecolor=fig.get_facecolor())


    def plot_features(self, filename):

        eval = pd.read_csv(filename + '.csv', index_col=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.patch.set_facecolor('#E0E0E0')

        ax.hist(eval['Features changed']/19, bins=100)
        plt.title('Features changed')
        plt.savefig(filename + '_f.png', facecolor=fig.get_facecolor())

    def plot_dist_dp(self, filename):

        eval = pd.read_csv(filename + '.csv', index_col=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.patch.set_facecolor('#E0E0E0')
        
        print('Dist to profile: {}'.format(np.mean(eval['Distance to data point'])))

        ax.hist(eval['Distance to data point'], bins=100)
        plt.title('Distance to the current data point')
        plt.savefig(filename + '_dp.png', facecolor=fig.get_facecolor())

    def plot_dist_td(self, filename):

        eval = pd.read_csv(filename + '.csv', index_col=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.patch.set_facecolor('#E0E0E0')
        
        print('Dist to training point: {}'.format(np.mean(eval['Distance to training data point'])))

        ax.hist(eval['Distance to training data point'], bins=100)
        plt.title('Distance to the nearest training data point')
        plt.savefig(filename + '_td.png', facecolor=fig.get_facecolor())

    def plot_bad_shap(self, filename):

        eval = pd.read_csv(filename + '.csv', index_col=0)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.patch.set_facecolor('#E0E0E0')

        perc = eval['Perc. features changed with bad shap vals']

        print('perc bad shap Features changed avg: {}'.format((np.mean(perc))))

        ax.hist(perc, bins=100)
        plt.title('Perc. changed features with contradictory SHAP changes')
        plt.savefig(filename + '_bs.png', facecolor=fig.get_facecolor())

    def print_target_missed(self, filename):
        eval = pd.read_csv(filename + '.csv', index_col=0)
        target_missed = eval['Target missed']
        print('target missed: {}'.format(np.sum(target_missed)))

    def print_realism_score(self, filename):
        eval = pd.read_csv(filename + '.csv', index_col=0)
        target_missed = eval['Realism score']
        print('realism score: {}'.format(np.mean(target_missed)))
        
        