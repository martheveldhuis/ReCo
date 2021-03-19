import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler

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
        :param name: string of the file name.
        :param feature_names: list of strings with feature names.
        :param outcome_name: string of outcome feature name.
        :param test_size: proportion of test set (optional: default is 0.2).
        """
        # Turn off matplotlib interactive mode to prevent it from popping up.
        plt.ioff()

        if isinstance(params['data'], pd.DataFrame):
            self.data = params['data']
        else:
            raise ValueError('should provide data in a pandas dataframe')

        if type(params['name']) is str:
            self.file_name = params['name']
        else:
            raise ValueError('should provide the name of the file')

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
        self.scaled_train_data = self.scale_data(self.train_data)
        

    def split_data(self, data):
        """Split the data into training and testing datasets, stratify on outcome"""

        train_df, test_df = train_test_split(data, test_size=self.test_size, 
                                             random_state=0, 
                                             stratify=data[self.outcome_name])

        # print("Splitting data into training with ", train_df.shape, "sampes and ",
        #       test_df.shape, "testing samples")

        return train_df, test_df

    def scale_data(self, train_data):
        """Scale and translate each feature individually such that it is between zero and one."""

        # Fit on training data only.
        # scaler = StandardScaler().fit(train_data[self.feature_names])
        scaler = QuantileTransformer().fit(train_data[self.feature_names])
        self.scaler = scaler
        scaled_train_data = scaler.transform(train_data[self.feature_names])

        scaled_train_data_df = pd.DataFrame(data=scaled_train_data, columns=self.feature_names)
        scaled_train_data_df.index = train_data.index
        scaled_train_data_df[self.outcome_name] = train_data[self.outcome_name]
        print(train_data['D1S1656 perc. known alleles'].max())

        return scaled_train_data_df

    def scale_data_point(self, data_point):
        """Translate 1 data point such that all its feature values are between zero and one."""
        
        data_point_scaled = pd.Series(self.scaler.transform(data_point[self.feature_names].to_numpy().reshape(1, -1)).ravel())
        data_point_scaled.name = data_point.name
        data_point_scaled.index = self.feature_names
        
        # Set any values > 1 to 1. This is only used in visualization.
        data_point_scaled = data_point_scaled.where(data_point_scaled <= 1.0, 1.0)
        #data_point_scaled.values = data_point_scaled.values.apply(> 1.0 else 1.0 for y in x])

        return data_point_scaled

    def get_features_min_max(self):
        """
            Generate a dataframe with min and max values of each feature.
            Note that we take these values from the entire dataset we have available
            If values 

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
            med = np.median(self.train_data[feature_name].values)
            # Compute the deviation from median for each feature value.
            dev_from_med = abs(self.train_data[feature_name].values - med)
            # The MAD = the median of the deviations.
            mad = np.median(dev_from_med)
            # To avoid division by zero, and any negative values.
            if mad <= 0.0:
                mad = 1.0
            mads[feature_name] = mad
        
        return mads

    def plot_feature_correlations(self):
        """Plot the pairwise feature correlations."""

        fig = plt.figure(figsize=(18,18))
        fig.patch.set_facecolor('#E0E0E0')
        sns.heatmap(self.train_data.astype(float).corr(method='kendall'), linewidths=0.1, vmin=-1.0,
                    vmax=1.0, square=True, linecolor='white', annot=True, 
                    cmap="PiYG")
        plt.savefig(r'data_analysis\correlations_kendall_' + self.file_name + '.png', facecolor=fig.get_facecolor())

    def plot_feature_boxplots(self):
        """Plot boxplot for each feature."""

        num_features = len(self.feature_names)
        fig, ax = plt.subplots(nrows=1, ncols=num_features, figsize=(50, 2), tight_layout=True)
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(num_features):
            ax[i].boxplot(self.train_data[self.train_data.columns[i]])
            ax[i].set_xlabel(self.train_data.columns[i])

        plt.savefig(r'data_analysis\boxplots_' + self.file_name + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')

    def plot_feature_histograms(self):
        """Plot histogram for each feature."""

        num_features = len(self.feature_names)
        fig, ax = plt.subplots(nrows=1, ncols=num_features, figsize=(50, 2))
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(num_features):
            ax[i].hist(self.train_data[self.train_data.columns[i]], bins=50)
            ax[i].set_xlabel(self.train_data.columns[i])

        plt.savefig(r'data_analysis\histograms_' + self.file_name + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')

    def plot_feature_violin(self):
        """Plot violin plots for each feature."""

        num_features = len(self.feature_names)
        fig, ax = plt.subplots(nrows=1, ncols=num_features, figsize=(50, 2))
        fig.patch.set_facecolor('#E0E0E0')

        for i in range(num_features):
            ax[i].violinplot(self.scaled_train_data[self.train_data.columns[i]], widths=0.9,
                        showmeans=False, showextrema=False, showmedians=False)
            ax[i].set_xlabel(self.train_data.columns[i])

        plt.savefig(r'data_analysis\violins_' + self.file_name + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')

    def plot_lda_3d(self):
        """Generate a gif from LDA"""

        # Initialize LDA in 3D, fit on data.
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda_data = lda.fit(self.train_data[self.feature_names], 
                           self.train_data[self.outcome_name]).transform(self.train_data[self.feature_names])
        print("Explained variance with 3 components: ", lda.explained_variance_ratio_)

        # Label based on outcome (1-5).
        lda_x = lda_data[:,0]
        lda_y = lda_data[:,1]
        lda_z = lda_data[:,2]
        labels = self.train_data[self.outcome_name].values
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
        ani.save(r'data_analysis\lda_' + self.file_name + '.gif', writer=animation.PillowWriter(fps=20))  

    def plot_lda_2d(self):
        """Generate an LDA plot"""

        # Initialize LDA in 3D, fit on data.
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda_data = lda.fit(self.train_data[self.feature_names], 
                           self.train_data[self.outcome_name]).transform(self.train_data[self.feature_names])
        print("Explained variance with 2 components: ", lda.explained_variance_ratio_)

        # Label based on outcome (1-5).
        lda_x = lda_data[:,0]
        lda_y = lda_data[:,1]
        labels = self.train_data[self.outcome_name].values
        colors = {1:'tab:purple', 2:'tab:orange', 3:'tab:green', 4:'tab:red', 5:'tab:blue'}
  
        # Define the graph.
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot()

        for l in np.unique(labels):
            i = np.where(labels==l)
            ax.scatter(lda_x[i], lda_y[i], c=colors[l], alpha=0.1)

        plt.xlabel("LDA 1",fontsize=14)
        plt.ylabel("LDA 2",fontsize=14)
        
        plt.savefig(r'data_analysis\lda2d_' + self.file_name + '.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        return lda
