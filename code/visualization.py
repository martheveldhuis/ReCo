import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Visualization:

    def __init__(self, data_point, data_point_scaled, prediction, shap_values):
        """Init method

        :param data_point: pandas series of the data point we are visualizing.
        :param data_point_scaled: pandas series of the scaled data point we are visualizing.
        :param prediction: float for the data point prediction.
        :param shap_values: array of floats representing the SHAP values of each feature.

        """
        if isinstance(data_point, pd.Series):
            self.data_point = data_point
        else: 
            raise ValueError("should provide data point in a pandas series")

        if isinstance(data_point_scaled, pd.Series):
            self.data_point_scaled = data_point_scaled
        else: 
            raise ValueError("should provide scaled data point in a pandas series")

        if type(prediction) is float or isinstance(prediction, np.float64):
            self.prediction = prediction
        else:
            raise ValueError("should provide data point prediction as a float")

        if isinstance(shap_values, np.ndarray):
            self.shap_values = shap_values
        else:
            raise ValueError("should provide data point shap values as a float array")

        # Create the figure skeleton, plot shap values, plot the profile feature values.
        self.fig, self.shap_ax, self.profile_ax, self.colorbar_ax = self.create_figure()
        self.plot_shap()
        self.plot_profile()
        self.fig.savefig(r'results\new_' + self.data_point.name + '.png', 
                        facecolor=self.fig.get_facecolor(), bbox_inches='tight')

    def create_figure(self):
        """Creates a framework for the entire visualization"""

        fig, axs = plt.subplots(2,2, figsize=(18,9), 
                                gridspec_kw=dict(height_ratios=[19,1], width_ratios=[1,2]), 
                                tight_layout=True, constrained_layout=False)
        plt.subplots_adjust(hspace=0.0, wspace=0.1)
        fig.suptitle('Profile ' + self.data_point.name + 
                    ' was predicted to have {:.2f}'.format(self.prediction) +
                    ' ({}) contributors.'.format(round(self.prediction)) +
                    ' Below you will find the top features for the current prediction (left)' +
                    ' and the feature values on a normalized scale (right)', 
                    fontsize=14, ha='center')
        fig.facecolor = 'w'

        # Define which axis are which.
        shap_ax = axs[0,0]
        profile_ax = axs[0,1]
        colorbar_ax = axs[1,0]
        axs[1,1].set_visible(False)

        return fig, shap_ax, profile_ax, colorbar_ax

    def plot_shap(self):
        """Creates the left-most figure containing the SHAP values for the current prediction."""
        
        # Create a local axis to avoid calling the class axis frequently.
        ax = self.shap_ax

        # Create a colormap with 0 as the center, offset our shap values to this.
        colormap = plt.get_cmap('coolwarm_r')
        offset = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0., vmax=1.0)
        colors = offset(self.shap_values)  

        # Plot the SHAP values per feature, colored by colormap
        ax.barh(self.data_point.index, self.shap_values, color=colormap(colors))
        
        # Put first feature at the top, add a zero-line, set y labels to have feature values, 
        # remove x labels, set a fixed x-axis, and title.
        ax.invert_yaxis()
        ax.axvline(x=0, c='#E0E0E0')
        y_labels = [(self.data_point.index[i] + ' = {:.4g}').format(self.data_point.iloc[i]) for i in range(len(self.data_point.index))]
        ax.set_yticks(ax.get_yticks()) # MATPLOTLIB 3.3.1 BUG: it's only here to avoid a warning.
        ax.set_yticklabels(y_labels)
        ax.set_xticklabels([])
        ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_title('The following features values have contributed most to the current prediction', 
                     loc='right', fontsize=12)

        # Create a color bar legend to serve as our x-axis tick labels
        colorbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=offset, cmap=colormap), 
                                     cax=self.colorbar_ax, orientation='horizontal', pad=0.01)
        colorbar.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
        cb_labels = ['\N{MINUS SIGN}1.0', '\N{MINUS SIGN}0.8','\N{MINUS SIGN}0.6',
                     '\N{MINUS SIGN}0.4', '\N{MINUS SIGN}0.2', '0.0', 
                     '+0.2', '+0.4', '+0.6', '+0.8', '+1.0']
        colorbar.set_ticklabels(cb_labels)

        # Set local ax back to class ax
        self.shap_ax = ax

    def plot_profile(self):
        """Creates the right figure containing just the feature values of the profile."""

        # Create local variables.
        dp = self.data_point
        dp_scaled = self.data_point_scaled
        ax = self.profile_ax

        # Plot the bars representing the feature values.
        ax.barh(dp.index, dp_scaled, alpha=1, color='#E0E0E0', edgecolor='#E0E0E0')
        ax.set_xticklabels([]) # remove x-values
        ax.set_yticklabels([]) # remove y labels (they are in the shap plot)
        ax.set_xlim([0,1.08]) # ensure the scale is always the same, and fits the text (+0.08)
        ax.invert_yaxis() # put first feature at the top
        ax.set_title('These are all the feature values of this profile.', 
                                  loc='left', fontsize=12)
        
        # Add feature value at the end of the bar.
        for i, dp_bar in enumerate(ax.containers[0].get_children()):
            ax.text(dp_bar.get_width()+0.01, dp_bar.get_y()+dp_bar.get_height()/2.,
                    '{:.4g}'.format(dp[i]), ha='left', va='center')

        # Set local ax back to class ax
        self.profile_ax = ax

    def plot_counterfactual(self, counterfactual, counterfactual_scaled, counterfactual_pred, changes, tol=False):
        """Adjusts the right figure with a counterfactual visualization."""

        cf_ax = self.profile_ax
        data_point = self.data_point
        for i, dp_bar in enumerate(self.profile_ax.containers[0].get_children()):
            
            # Add changes to barchart only if the feature is changed in CF.
            dp_feature_name = data_point.index[i]
            if dp_feature_name in changes.index:

                # Grab x-coord of where to plot (dp scaled), and what number to display (cf val).
                dp_scaled_val = self.data_point_scaled[i]
                cf_feature_val = counterfactual[i]
                
                # Grab the original bar's y coordinate and height.
                bar_y = dp_bar.get_y()
                bar_height = dp_bar.get_height()

                diff = changes.loc[dp_feature_name].difference
                # Input value needs to be increased to match counterfactual.
                if diff > 0:
                    # Stack on top
                    cf_ax.barh(dp_feature_name, diff, left=dp_scaled_val,
                            color='w', alpha=1, edgecolor='#E0E0E0')
                    # Put data point value text inside bar.
                    cf_ax.texts[i].set_position((dp_bar.get_width()-0.01, 
                                                           bar_y+bar_height/2.))
                    cf_ax.texts[i].set_ha('right')                           
                    # Put cf data point value text outside bar.
                    cf_ax.text(dp_bar.get_width()+diff+0.01, bar_y+bar_height/2.,
                                         '{:.4g}'.format(cf_feature_val), ha='left', va='center')
                    # Add arrow
                    head_length = 0.01 if diff >= 0.01 else 0.75 * diff
                    cf_ax.arrow(dp_scaled_val, bar_y+bar_height/2., diff, 0, width=0.1, 
                                          color='tab:olive', length_includes_head=True, 
                                          head_width=0.4, head_length=head_length)
                # Input value needs to be decreased to match counterfactual.
                else:
                    cf_ax.barh(dp_feature_name, abs(diff), left=dp_scaled_val+diff,
                            color=dp_bar.get_facecolor(), alpha=1, edgecolor='w')
                    # Put cf data point value text inside bar.
                    cf_ax.text(dp_bar.get_width()+diff-0.01, bar_y+bar_height/2.,
                            '{:.4g}'.format(cf_feature_val), ha='right', va='center')
                    # Add arrow
                    head_length = 0.01 if abs(diff) >= 0.01 else 0.75 * abs(diff)
                    cf_ax.arrow(dp_scaled_val, bar_y+bar_height/2., diff, 0, width=0.1, 
                                          color='tab:purple', length_includes_head=True, 
                                          head_width=0.4, head_length=head_length)

        self.fig.suptitle('Profile ' + self.data_point.name + 
                          ' was predicted to have {:.2f}'.format(self.prediction) +
                          ' ({}) contributors.'.format(round(self.prediction)) + 
                          ' Below you will find the top features for the current prediction (left)' +
                          ' and changes to reach an alternative prediction (right)', 
                          fontsize=14, ha='center')


        if(tol):
            cf_ax.set_title('The following changes in feature values show the top changes ' +
                            'needed to reach the prediction of {}'.format(round(counterfactual_pred)) +
                            ' contributors.', loc='left', fontsize=12)

        else:
            cf_ax.set_title('The following changes in feature values show the most similar ' +
                            'profile with a prediction of {}'.format(round(counterfactual_pred)) +
                            ' contributors.', loc='left', fontsize=12)

        self.profile_ax = cf_ax

        if(tol):
            self.fig.savefig(r'results\new_' + data_point.name + '_' + counterfactual.name + '_tol.png', 
                    facecolor=self.fig.get_facecolor(), bbox_inches='tight')
        else:
            self.fig.savefig(r'results\new_' + data_point.name + '_' + counterfactual.name + '.png', 
                    facecolor=self.fig.get_facecolor(), bbox_inches='tight')


    def plot_counterfactual_tol(self, counterfactual, counterfactual_scaled, counterfactual_pred, changes):
        
        self.profile_ax.clear()
        self.plot_profile()
        self.plot_counterfactual(counterfactual, counterfactual_scaled, counterfactual_pred, changes, True)