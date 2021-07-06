import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19

with open('file_paths.json') as json_file:
    file_paths = json.load(json_file)
file_path = file_paths['original_590']
test_fraction = 0.2
data_reader = DataReader19Features(file_path, test_fraction)

dataset = Dataset(data_reader.read_data())

rf_classifier = RFC19(dataset, 'RFC19.sav')
rf_regressor = RFR19(dataset, 'RFR19.sav')

def create_conf_matrix(y_t, y_p, name, test=False):
    conf_mat = confusion_matrix(y_t, y_p, labels=[1.0,2.0,3.0,4.0,5.0])
    sns.heatmap(conf_mat, annot=True, cmap='Blues', 
                xticklabels=[1.0,2.0,3.0,4.0,5.0], yticklabels=[1.0,2.0,3.0,4.0,5.0])
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
create_conf_matrix(dataset.train_data[dataset.outcome_name], 
                    rf_classifier.train_pred, rf_classifier.model_name)
create_conf_matrix(dataset.train_data[dataset.outcome_name], 
                    rf_regressor.train_pred.round(), rf_regressor.model_name)

# Testing data
create_conf_matrix(dataset.test_data[dataset.outcome_name], 
                    rf_classifier.get_prediction(dataset.test_data[dataset.feature_names]), 
                    rf_classifier.model_name, test=True)
create_conf_matrix(dataset.test_data[dataset.outcome_name], 
                    rf_regressor.get_prediction(dataset.test_data[dataset.feature_names]).round(), 
                    rf_regressor.model_name, test=True)
