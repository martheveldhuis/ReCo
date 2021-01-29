import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

class DataReader:
    """An interface for data reader behaviour."""

    def read_data(self, file_path):
        """Interface for data reading behavior, must be implemented for specific data reading.

        :param file_path: string of the file path to the data.
        :return: a dictionary of params with keys as specified in Dataset()

        """
        raise NotImplementedError('should be overridden with specific data reader')

class Dataset:
    """A class that holds the data, must be generated using a DataReader implementation."""

    def __init__(self, params):
        """Init method

        dictionary containing all parameters.
        :param data: Pandas DataFrame with column names as feature names, row names as profiles.
        :param feature_names: list of strings with feature names.
        :param outcome_name: string of outcome feature name.
        :param test_size: proportion of test set (optional: default is 0.2).
        """

        if isinstance(params['data'], pd.DataFrame):
            self.data = params['data']
        else:
            raise ValueError('should provide data in a pandas dataframe')

        if type(params['feature_names']) is list:
            self.feature_names = params['feature_names']
        else:
            raise ValueError('should provide the names of the data features')

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError('should provide the name of outcome feature')

        if type(params['test_size']) is float:
            self.test_size = params['test_size']
        else:
            self.test_size = 0.2

        self.train_data, self.test_data = self.split_data(self.data)

    def split_data(self, data):
        """Split the data into training and testing datasets, stratify on outcome"""

        train_df, test_df = train_test_split(data, test_size=self.test_size, 
                                             random_state=0, 
                                             stratify=data[self.outcome_name])

        # print("Splitting data into training with ", train_df.shape, "sampes and ",
        #       test_df.shape, "testing samples")

        return train_df, test_df

    def get_features_min_max(self):
        """Generate a dataframe with min and max values of each feature.

            :return dataframe of min and max values per feature.
        """
        min_max_list = []

        # Get each feature's min and max values.
        for feature_name in self.feature_names:
            min = self.data[feature_name].min()
            max = self.data[feature_name].max()
            min_max_list.append([min, max])

        # Create dataframe from list of lists in correct format
        min_max_df = pd.DataFrame(min_max_list)
        min_max = min_max_df.T
        min_max.columns = self.feature_names
        min_max.index = ['min', 'max']

        return min_max

    def get_features_mad(self):
        """Compute Median Absolute Deviation of features."""

        mads = {}

        for feature_name in self.feature_names:
            med = np.median(self.data[feature_name].values)
            # Compute the deviation from median for each feature value.
            dev_from_med = abs(self.data[feature_name].values - med)
            # The MAD = the median of the deviations.
            mad = np.median(dev_from_med)
            # To avoid division by zero, and any negative values.
            if mad <= 0.0:
                mad = 1.0
            mads[feature_name] = mad
        
        return mads

    def plot_feature_correlations(self):
        """Plot the pairwise feature correlations."""

        fig = plt.figure(figsize=(15,15))
        fig.patch.set_facecolor('#E0E0E0')
        sns.heatmap(self.data.astype(float).corr(), linewidths=0.1, vmin=-1.0,
                    vmax=1.0, square=True, linecolor='white', annot=True, 
                    cmap="PiYG")
        plt.savefig("correlations.png", facecolor=fig.get_facecolor())
        plt.show()

    def plot_feature_boxplots(self):
        """Plot boxplot for each feature."""

        num_features = len(self.feature_names)
        fig, ax = plt.subplots(nrows=1, ncols=num_features, figsize=(50, 2), tight_layout=True)
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(num_features):
            ax[i].boxplot(self.data[self.data.columns[i]])
            ax[i].set_xlabel(self.data.columns[i])

        plt.savefig("boxplots.png", facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.show()

    def plot_feature_histograms(self):
        """Plot histogram for each feature."""

        num_features = len(self.feature_names)
        fig, ax = plt.subplots(nrows=1, ncols=num_features, figsize=(50, 2))
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(num_features):
            ax[i].hist(self.data[self.data.columns[i]], bins=50)
            ax[i].set_xlabel(self.data.columns[i])

        plt.savefig("histograms.png", facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.show()

    def plot_lda(self):
        """Generate a gif from LDA"""

        # Initialize LDA in 3D, fit on data.
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda_data = lda.fit(self.data[self.feature_names], 
                           self.data[self.outcome_name]).transform(self.data[self.feature_names])
        print("Explained variance with 3 components: ", lda.explained_variance_ratio_)

        # Label based on outcome (1-5).
        lda_x = lda_data[:,0]
        lda_y = lda_data[:,1]
        lda_z = lda_data[:,2]
        labels = self.data[self.outcome_name].values
        colors = {1:'tab:purple', 2:'tab:orange', 3:'tab:green', 4:'tab:red', 5:'tab:blue'}
  
        # Define the graph.
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(projection='3d')

        for l in np.unique(labels):
            i = np.where(labels==l)
            ax.scatter(lda_x[i], lda_y[i], lda_z[i], c=colors[l])

        plt.xlabel("LDA 1",fontsize=14)
        plt.ylabel("LDA 2",fontsize=14)
        
        # Define the animation.
        def rotate(angle):
            ax.view_init(azim=angle)

        angle = 1
        ani = animation.FuncAnimation(fig, rotate, 
                                      frames=np.arange(0, 360, angle), 
                                      interval=50)
        ani.save('lda.gif', writer=animation.PillowWriter(fps=20))  
