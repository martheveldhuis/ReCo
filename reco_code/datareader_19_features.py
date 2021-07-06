import os
import json
import pandas as pd
from data import DataReader

class DataReader19Features(DataReader):
    """Implementation of DataReader for the files containing the 19 features."""

    def __init__(self, file_path, test_size):
        """Init method

        :param file_path: file path containing a file of data.
        :param test_size (optional): fraction of the data to be used for testing.
        """

        if os.path.isfile(file_path):
            self.file_path = file_path
        else:
            raise OSError('file path provided does not have a file')

        if type(test_size) is float:
            self.test_size = test_size
        else:
            self.test_size = 0.2

    def read_data(self, file_path=0):
        """Extracts 19 features from the specified file for each profile."""

        if file_path == 0:
            file_path = self.file_path

        # Read comma-separated file with our 19 features.
        data = pd.read_csv(file_path, index_col=0)

        # The sample name (index) + 19 features + NOC
        features = ['index','MAC', 'TAC', 'MinNOC_CSF1PO', 'MinNOC_D16S539', 
                    'PercAF_D1S1656', 'AlleleCount_D3S1358', 
                    'AlleleCount_D8S1179', 'MinNOC_Penta D', 'MinNOC_Penta E', 
                    'SumAF_TH01', 'AlleleCount_TPOX', 'MinNOC_TPOX', 
                    'stdHeight_vWA', 'stdAllele', 'MAC0', 'MAC5-6', 
                    'peaksBelowRFU', 'MatchProbability', 'MinNOC', 'NOC']

        # Extract the data file name (we assume name to be between last \ and .)
        path = self.file_path
        dot = path.rfind('.')
        last_slash = path.rfind('\\')
        name = path[last_slash+1:dot]

        # Translate features to more understandable strings.
        with open('feature_translations.json') as json_file:
            translations = json.load(json_file)
        data.rename(columns = translations, inplace = True)

        # Grab the outcome- and feature names.
        outcome_name = 'NOC'
        feature_names = data.columns[~data.columns.isin([outcome_name])].values.tolist()

        # Make sure the features are all numeric.
        data[feature_names] = data[feature_names].astype(float)

        # Create the dictionary object to pass to data.
        params = { 'data' : data, 
                   'name' : name,
                   'feature_names' : feature_names, 
                   'outcome_name' : outcome_name, 
                   'test_size' : self.test_size }

        return params
