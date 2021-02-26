# random forest for feature importance on a regression problem
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data import Dataset
from datareader_278_features import DataReader278Features

file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_278.txt'
test_fraction = 0.2
data_reader_278 = DataReader278Features(file_path, test_fraction)
dataset_278 = Dataset(data_reader_278.read_data())

# define dataset
X = dataset_278.train_data[dataset_278.feature_names]
y = dataset_278.train_data[dataset_278.outcome_name]
# define the model
model = RandomForestRegressor()
# fit the model

zeros=np.zeros(shape=(276,2))
importances = pd.DataFrame(data=zeros, columns=['importance', 'avg'], index=dataset_278.feature_names)
for i in range(0,9):
    model.fit(X, y)
    # get importance
    for j,v in enumerate(model.feature_importances_):
        importances['importance'].iloc[j]+=v

avgs = []
for i,im in importances.iterrows():
    avgs.append(im[0]/10)

importances['avg'] = avgs
sorted_importances = importances.sort_values(by=['avg'], ascending=False)
print(sorted_importances.head(25))