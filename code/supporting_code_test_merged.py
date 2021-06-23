import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from data import Dataset
from datareader_19_features import DataReader19Features
from sklearn_predictors import RFC19
from sklearn_predictors import RFR19

file_path = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt'
test_fraction = 0.2
data_reader = DataReader19Features(file_path, test_fraction)

dataset = Dataset(data_reader.read_data())

rf_classifier = RFC19(dataset, 'RFC19_merged.sav')
rf_regressor = RFR19(dataset, 'RFR19_merged.sav')

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

# # Training data  
# create_conf_matrix(dataset.train_data[dataset.outcome_name], rf_classifier.train_pred, rf_classifier.model_name)
# create_conf_matrix(dataset.train_data[dataset.outcome_name], rf_regressor.train_pred.round(), rf_regressor.model_name)

# # Testing data
# create_conf_matrix(dataset.test_data[dataset.outcome_name], rf_classifier.get_prediction(dataset.test_data[dataset.feature_names]), rf_classifier.model_name, test=True)
# create_conf_matrix(dataset.test_data[dataset.outcome_name], , rf_regressor.model_name, test=True)

# correct_pred = []
# incorrect_pred = []
# test_pred = rf_regressor.get_prediction(dataset.test_data[dataset.feature_names])
# for i in range(len(test_pred)):
#     if test_pred[i].round() == dataset.test_data[dataset.outcome_name].iloc[i]:
#         correct_pred.append(i)
#     else:
#         incorrect_pred.append(i)
# correct_preds = dataset.test_data.iloc[correct_pred, :]
# incorrect_preds = dataset.test_data.iloc[incorrect_pred, :] 
# print(incorrect_preds.shape)

# for profile, row in incorrect_preds.iterrows():
#     index = rf_regressor.dataset.test_data.index.get_loc(profile)
#     pred = test_pred[index]
#     actual = rf_regressor.dataset.test_data.loc[profile]['NOC']
#     print(profile + ';{}'.format(pred) + ';{}'.format(actual))


# correct_pred = []
# incorrect_pred = []
# test_pred = rf_classifier.get_prediction(dataset.test_data[dataset.feature_names])
# for i in range(len(test_pred)):
#     if test_pred[i].round() == dataset.test_data[dataset.outcome_name].iloc[i]:
#         correct_pred.append(i)
#     else:
#         incorrect_pred.append(i)
# correct_preds = dataset.test_data.iloc[correct_pred, :]
# incorrect_preds = dataset.test_data.iloc[incorrect_pred, :] 
# print(incorrect_preds.shape)
# print(correct_preds.shape)

# for profile, row in incorrect_preds.iterrows():
#     index = rf_classifier.dataset.test_data.index.get_loc(profile)
#     pred = test_pred[index]
#     actual = rf_classifier.dataset.test_data.loc[profile]['NOC']
#     print(profile + ';{}'.format(pred) + ';{}'.format(actual))


# print(dataset.data.shape)
# print(dataset.test_data.shape)

# original_in_test = 0
# for profile, row in dataset.test_data.iterrows():
#     if 'Run 1_Trace' in profile:
#         continue
#     else:
#         original_in_test+=1
# print(original_in_test)

# test_pred_cl = rf_classifier.get_prediction(dataset.test_data[dataset.feature_names])

# acc_r = accuracy_score(dataset.test_data[dataset.outcome_name].to_numpy(), test_pred.round())
# acc_c = accuracy_score(dataset.test_data[dataset.outcome_name].to_numpy(), test_pred_cl)

# print('acc reg: {}'.format(acc_r))
# print('acc cl: {}'.format(acc_c))

# train_pred = rf_classifier.get_prediction(dataset.train_data[dataset.feature_names])
# for profile, row in rf_classifier.incorrect_pred.iterrows():
#     index = rf_classifier.dataset.train_data.index.get_loc(profile)
#     pred = train_pred[index]
#     actual = rf_classifier.dataset.train_data.loc[profile]['NOC']
#     print(profile + ';{}'.format(pred) + ';{}'.format(actual))

print(rf_classifier.incorrect_pred.shape)