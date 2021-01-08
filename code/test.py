import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import sklearn
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class DataReader:
    file_path = None
    X = None
    y = None

    def __init__(self, file_path):
        self.file_path = file_path
        print("Initalizing data reader with file at: ", file_path)

    def get_data(self):
        return self.read_data

    def read_data(self):
        "Must be implemented for each type of file"
        raise NotImplementedError



class DataAnalyzer:
    file_path = r"D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590.txt"
    # The sample name (index) + 19 features (should be provided in file) + NOC
    features = ['index','MAC', 'TAC', 'MinNOC_CSF1PO', 'MinNOC_D16S539', 
                'PercAF_D1S1656', 'AlleleCount_D3S1358', 'AlleleCount_D8S1179',
                'MinNOC_Penta D', 'MinNOC_Penta E', 'SumAF_TH01', 
                'AlleleCount_TPOX', 'MinNOC_TPOX', 'stdHeight_vWA', 
                'stdAllele', 'MAC0', 'MAC5-6', 'peaksBelowRFU', 'MatchKans', 
                'MinNOC', 'NOC']
    X = None
    y = None

    def read_file(self):
        with open(self.file_path) as json_file:
            dictionary = json.load(json_file)

        # Read flattened file
        raw_df = pd.DataFrame.from_dict(dictionary, orient='index')
        raw_df.reset_index(level=0, inplace=True)
        json_struct = json.loads(raw_df.to_json(orient="records"))  
        flat_df = pd.json_normalize(json_struct) 

        # Change "Locus.TPOX.SumAF_TPOX" to "SumAF_TPOX" to match features
        new_column_names = []
        for column_name in flat_df.columns: 
            if "Locus" in column_name:
                new_column_name = column_name[column_name.rfind(".")+1:]
            else:
                new_column_name = column_name
            new_column_names.append(new_column_name) 
        flat_df.columns = new_column_names

        # Filter features to only include the ones specified in features
        data = flat_df[self.features]    
        data.set_index('index', inplace=True)
        data.index.name = None
        
        # Split X from y.
        self.X = data.drop('NOC', axis=1)
        self.y = data['NOC'].astype(float)

    def plot_feature_correlations(self):
        plt.figure(figsize=(40,20))
        sns.heatmap(self.X.astype(float).corr(),linewidths=0.1,vmax=1.0, 
                    square=True, linecolor='white', annot=True)
        plt.savefig("correlations.png")
        plt.show()

    def plot_lda(self):
        lda = LinearDiscriminantAnalysis(n_components=3)
        X_r2 = lda.fit(self.X, self.y).transform(self.X)
        print("Explained variance with 3 components: ", 
              lda.explained_variance_ratio_)

        lda_x = X_r2[:,0]
        lda_y = X_r2[:,1]
        lda_z = X_r2[:,2]
        labels = self.y
        colors = {1:'red', 2:'green', 3:'blue', 4:'magenta', 5:'yellow'}
  
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(projection='3d')

        for l in np.unique(labels):
            i = np.where(labels==l)
            ax.scatter(lda_x[i], lda_y[i], lda_z[i], c=colors[l])

        plt.xlabel("LDA 1",fontsize=14)
        plt.ylabel("LDA 2",fontsize=14)
        
        def rotate(angle):
            ax.view_init(azim=angle)

        angle = 1
        ani = animation.FuncAnimation(fig, rotate, 
                                      frames=np.arange(0, 360, angle), 
                                      interval=50)
        ani.save('lda.gif', writer=animation.PillowWriter(fps=20))





#data_analyzer.plot_feature_correlations()
#data_analyzer.plot_lda()