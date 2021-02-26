import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19

file_path_original = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19_original.txt'
file_path_edited = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19.txt'
test_fraction = 0.2
data_reader_original = DataReader19Features(file_path_original, test_fraction)
data_reader_edited = DataReader19Features(file_path_edited, test_fraction)

dataset_original = Dataset(data_reader_original.read_data())
dataset_edited = Dataset(data_reader_edited.read_data())

rf_classifier = RFC19(dataset_original, 'RFC19.sav')
rf_regressor = RFR19(dataset_edited, 'RFR19.sav')

# for profile, row in rf_regressor.incorrect_pred.iterrows():
#     #print(row)
#     index = rf_regressor.dataset.train_data.index.get_loc(profile)
#     pred = rf_regressor.train_pred[index]
#     actual = rf_regressor.dataset.train_data.loc[profile]['NOC']
#     print('Predicted as: {}'.format(pred))
#     print('Actual NOC: {}'.format(actual))

# print('Number of wrong predictions is {}'.format(len(rf_classifier.incorrect_pred.index)))
# for profile, row in rf_classifier.incorrect_pred.iterrows():
#     #print(row)
#     index = rf_classifier.dataset.train_data.index.get_loc(profile)
#     pred = rf_classifier.train_pred[index]
#     actual = rf_classifier.dataset.train_data.loc[profile]['NOC']
#     print('Predicted as: {}'.format(pred))
#     print('Actual NOC: {}'.format(actual))

def create_conf_matrix(y_t, y_p, name, test=False):
    conf_mat = confusion_matrix(y_t, y_p, labels=[1.0,2.0,3.0,4.0,5.0])
    sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=[1.0,2.0,3.0,4.0,5.0], yticklabels=[1.0,2.0,3.0,4.0,5.0])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if test:
        plt.title('Confusion matrix testing data for model ' + name)
        plt.savefig('confusion_matrix_test' + name + '.png', dpi=100, bbox_inches='tight')
    else:
        plt.title('Confusion matrix training data for model ' + name)
        plt.savefig('confusion_matrix_train' + name + '.png', dpi=100, bbox_inches='tight')
    plt.close()

# Training data  
create_conf_matrix(dataset_original.train_data[dataset_original.outcome_name], rf_classifier.train_pred, rf_classifier.model_name)
create_conf_matrix(dataset_edited.train_data[dataset_edited.outcome_name], rf_regressor.train_pred.round(), rf_regressor.model_name)

# Testing data
create_conf_matrix(dataset_original.test_data[dataset_original.outcome_name], rf_classifier.get_prediction(dataset_original.test_data[dataset_original.feature_names]), rf_classifier.model_name, test=True)
create_conf_matrix(dataset_edited.test_data[dataset_edited.outcome_name], rf_regressor.get_prediction(dataset_edited.test_data[dataset_edited.feature_names]).round(), rf_regressor.model_name, test=True)
