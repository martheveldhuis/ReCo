# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from anchor import utils
# from anchor import anchor_tabular
# from data import Dataset
# from predictor import Predictor

# class AnchorsGenerator:
#     """A class for generating Anchors."""

#     def __init__(self, dataset, predictor):
#         """Init method

#         :param dataset: Dataset instance containing all data information.
#         :param predictor: Predictor instance wrapping all predictor functionality.
#         """

#         if isinstance(dataset, Dataset):
#             self.dataset = dataset
#         else:
#             raise ValueError("should provide data as Dataset instance")

#         if isinstance(predictor, Predictor):
#             self.predictor = predictor
#         else:
#             raise ValueError("should provide predictor as Predictor instance")

#         self.explainer = self.create_explainer()

#     def create_explainer(self):

#         # Format classnames and training data
#         class_names = self.dataset.data[self.dataset.outcome_name].unique().astype(int).astype(str).tolist()
#         feature_names = self.dataset.feature_names
#         train_data_np = self.dataset.train_data[self.dataset.feature_names].to_numpy()

#         # Create the explainer object
#         explainer = anchor_tabular.AnchorTabularExplainer(class_names,
#                                                           feature_names, 
#                                                           train_data_np, 
#                                                           discretizer='decile')

#         return explainer

#     def generate_basic_anchor(self, data_point):

#         # Ensure we only use the features, not the outcome.
#         data_point_X = data_point[self.dataset.feature_names]
#         data_point_scaled = pd.Series(self.dataset.scaler.transform(data_point_X.to_numpy().reshape(1, -1)).ravel())

#         # Helper function required for Anchors.
#         def anchor_prediction(x):
#             # if self.predictor.get_prediction(x)[0] == 5.00:
#             #     print('this instance is 5: ')
#             #     print(x)
#             return self.predictor.get_prediction(x) # to deal with regression

#         # Generate the explanation.
#         data_point_np = data_point_X.to_numpy()
#         expl = self.explainer.explain_instance(data_point_np, anchor_prediction, threshold=0.9, max_anchor_size=6)

#         print(expl.examples(only_different_prediction=True, partial_index=0)[0])
#         anchor_plottable = self.translate_anchor(expl.features(), expl.names())

#         # Get the predictions.
#         top_pred, top_prob, second_pred, second_prob = self.predictor.get_top2_predictions(data_point_X)

#         # Create the Anchors object.
#         anchor = Anchor(data_point_X, data_point_scaled, top_pred, top_prob, self.predictor.model_name, 
#                         anchor_plottable, expl.names(), expl.precision(), expl.coverage())
#         return anchor
    
#     def generate_basic_cf_anchor(self, data_point):

#         # Ensure we only use the features, not the outcome.
#         data_point_X = data_point[self.dataset.feature_names]
#         scaled_data_point = self.dataset.scaler.transform(data_point_X)

#         # Helper function required for Anchors (will not actually be used here).
#         def anchor_prediction(x):
#             return self.predictor.get_prediction(x).round() # to deal with regression

#         # Get the predictions.
#         top_pred, top_prob, second_pred, second_prob = self.predictor.get_top2_predictions(data_point_X)

#         # Generate the explanation for the cf outcome.
#         data_point_np = data_point_X.to_numpy()
#         expl = self.explainer.explain_instance(data_point_np, anchor_prediction, threshold=0.9, 
#                                                desired_label=second_pred, max_anchor_size=6)

#         # Create the Anchors object.
#         anchor = Anchor(data_point_X, scaled_data_point, second_pred, second_prob, self.predictor.model_name, 
#                         expl.names(), expl.precision(), expl.coverage())
#         return anchor

#     def translate_anchor(self, feature_nums, feature_ranges):
#         """Translate each anchor rule to be plotted immediately."""
        
#         scaler = self.dataset.scaler
#         translated = dict.fromkeys(feature_nums, None) 

#         for feature_num, feature_range in zip(feature_nums, feature_ranges):
#             # Grab the min and max values for each feature.
#             min = scaler.data_min_[feature_num] 
#             max = scaler.data_max_[feature_num]
#             min_scaled = 0
#             max_scaled = 1

#             # Grab the values as stated in the anchor rules.
#             if '>' in feature_range:
#                 temp_min = float(feature_range.split('>')[1].strip())
#                 min_scaled = (temp_min - min) * scaler.scale_[feature_num]
#                 min = temp_min 
#             elif '< ' in feature_range:
#                 temp_min = float(feature_range.split('<')[0].strip())
#                 temp_max = float(feature_range.split('<=')[1].strip())
#                 min_scaled = (temp_min - min) * scaler.scale_[feature_num]
#                 max_scaled = (temp_max - min) * scaler.scale_[feature_num]
#                 min = temp_min
#                 max = temp_max
#             else:
#                 temp_max = float(feature_range.split('<=')[1].strip())
#                 max_scaled = (temp_max - min) * scaler.scale_[feature_num]
#                 max = temp_max
            
#             # Use the set min and max values per feature
#             translated[feature_num] = [float(min), float(min_scaled), 
#                                        float(max), float(max_scaled)]
            
#         return translated



# class Anchor:
#     """Simple wrapper class for Anchors"""

#     def __init__(self, data_point, data_point_scaled, pred, prob, model_name, anchor_plottable, feature_ranges, precision, coverage):
#         """Init method

#         :param data_point: pandas series of the data point we are explaining with this anchor.
#         :param data_point_scaled: pandas series of the scaled data point we are explaining with this anchor.
#         :param pred: float for the data point prediction.
#         :param prob: float for the probability of the data point prediction.
#         :param model_name: string representing the model.
#         :param anchor_plottable: ready to plot anchor.
#         :param feature_ranges: list of the feature ranges associated with an anchor.
#         :param precision: float representing the fraction of the instances which will be 
#                         predicted the same as the data point when this anchor holds.
#         :param coverage: float representing the probability of the anchor applying to 
#                         its perturbation space.
#         """

#         if isinstance(data_point, pd.Series):
#             self.data_point = data_point
#         else: 
#             raise ValueError("should provide data point in a pandas series")

#         if isinstance(data_point_scaled, pd.Series):
#             self.data_point_scaled = data_point_scaled
#         else: 
#             raise ValueError("should provide scaled data point in a pandas series")

#         if type(pred) is float or isinstance(pred, np.float64):
#             self.pred = pred
#         else:
#             raise ValueError("should provide data point prediction as a float")

#         if type(prob) is float or isinstance(prob, np.float64):
#             self.prob = prob
#         else:
#             raise ValueError("should provide data point prediction probability as a float")

#         if type(model_name) is str:
#             self.model_name = model_name
#         else:
#             raise ValueError("should provide model name as a string")

#         if type(anchor_plottable) is dict:
#             self.anchor_plottable = anchor_plottable
#         else:
#             raise ValueError("should provide plottable anchor as a dict")

#         if type(feature_ranges) is list:
#             self.feature_ranges = feature_ranges
#         else:
#             raise ValueError("should provide anchor feature ranges as list of strings")
        
#         if type(precision) is float or isinstance(precision, np.float64):
#             self.precision = precision
#         else:
#             raise ValueError("should provide anchor precision as a float")

#         if type(coverage) is float or type(coverage) is int or isinstance(coverage, np.float64):
#             self.coverage = coverage
#         else:
#             raise ValueError("should provide anchor coverage as a float")

#     def print_anchor_text(self):
#         """Simple print of anchor"""

#         print('\nProfile ' + self.data_point.name + 
#               ' was predicted by model ' + self.model_name + 
#               ' to have {}'.format(int(round(self.pred))) + # for regression
#               ' contributors, with a probability of {:.2f}'.format(self.prob) + '.')
#         print('The model will predict {}'.format(int(round(self.pred))) + # for regression
#               ' contributors {}'.format(int(self.precision*100)) + '% of the time' +
#               ' when ALL the following rules are true: \n%s ' % ' \n'.join(self.feature_ranges) +
#               '\nThese rules apply to original data with a probability of {:.2f}'.format(self.coverage))

#     def plot_anchor(self):
#         """Plot the anchor on top of the current profile"""

#         # Grab the values we need.
#         dp = self.data_point
#         dp_scaled = self.data_point_scaled
#         an = self.anchor_plottable
        
#         # Set up figure.
#         fig, ax = plt.subplots(figsize=(18,9))
#         #TODO: make title. fig.suptitle(self.get_title(), x=0.13, ha='left')
#         fig.patch.set_facecolor('#E0E0E0')

#         # Plot the bars for the scaled data point
#         ax.barh(dp.index, dp_scaled, color='w', alpha=1, edgecolor='tab:gray')
#         ax.set_xticklabels([]) # remove x-values
#         plt.gca().invert_yaxis() # put first feature at the top
#         labels = ax.get_yticklabels()

#         for i, dp_bar in enumerate(ax.containers[0].get_children()):
#             dp_feature_name = dp.index[i]
#             dp_feature_val = dp[i]
#             dp_scaled_val = dp_scaled[i]
            
#             bar_y = dp_bar.get_y()
#             bar_height = dp_bar.get_height()

#             # Add ranges to barchart.
#             if i in an.keys():
#                 # Grab the (scaled) anchor ranges.
#                 min, min_scaled, max, max_scaled = an[i]
#                 range = max_scaled - min_scaled

#                 # Plot the anchor range in cyan.
#                 ax.barh(dp_feature_name, range, left=min_scaled, 
#                         color='tab:cyan', alpha=0.5)
                
#                 # Check for overlapping text
#                 close_to_max = (abs(dp_scaled_val-max_scaled) < 0.06)
#                 max_correction = -0.01 if close_to_max else 0.01
#                 corrected_align = 'right' if close_to_max else 'left'

#                 # Plot the anchor max value.
#                 if max_scaled < 1: # don't plot 1
#                     ax.text(max_scaled+max_correction, bar_y+bar_height/2.,
#                             '<= {:.4g}'.format(max), ha=corrected_align, va='center', style='italic')
#                 # Plot the anchor min value.
#                 if min_scaled > 0: # don't plot 0
#                     ax.text(min_scaled-0.01, bar_y+bar_height/2.,
#                             '> {:.4g}'.format(min), ha='right', va='center', style='italic')
#                     # Some extra space for values close to 0.
#                     if min_scaled < 0.06:
#                         labels[i].set_x(labels[i].get_position()[0] - 0.06)

#             # Plot data point value outside bar.
#             ax.text(dp_bar.get_width()+0.01, bar_y+bar_height/2.,
#                     '{:.4g}'.format(dp_feature_val), ha='left', va='center')

    
        
#         # Save figure.
#         fig.tight_layout()
#         plt.savefig(r'results\anchor_v0' + self.model_name + '.png', 
#                     facecolor=fig.get_facecolor(), bbox_inches='tight')


# # ################################ ANCHORS ################################

# # Define Anchors generators (1 generator must be fitted to 1 predictor).
# # anchors_generator_c = AnchorsGenerator(dataset_merged, model)

# # Generate Anchors and print them.
# # anchor = anchors_generator_c.generate_basic_anchor(data_point)
# # anchor.plot_anchor()
# # anchor.print_anchor_text()