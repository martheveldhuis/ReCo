from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFR19
import dice_ml
from dice_ml.utils import helpers # helper functions

# Get data
file_path_merged = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt'
test_fraction = 0.2
data_reader_merged = DataReader19Features(file_path_merged, test_fraction)
dataset_merged = Dataset(data_reader_merged.read_data())


# Get model
model = RFR19(dataset_merged, 'RFR19_merged.sav')

# i = 0
# for index, row in dataset_merged.test_data.iterrows():
#     if index == '2.29':
#         print(i)
#     i+=1



continuous = []


d = dice_ml.Data(dataframe=dataset_merged.train_data, continuous_features=continuous, outcome_name=dataset_merged.outcome_name)
m = dice_ml.Model(model=model.model, backend='sklearn', model_type='regressor')
exp_random = dice_ml.Dice(d, m, method="random")
query_instances = dataset_merged.test_data[536:537][dataset_merged.feature_names]
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=3, desired_range=[2.5,3.4])
print(dataset_merged.test_data[536:537].T)
print(dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.iloc[0].T)


