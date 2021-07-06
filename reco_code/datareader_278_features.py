import os
import json
import pandas as pd
from data import DataReader

class DataReader278Features(DataReader):
    """Implementation of DataReader for the files containing all 278 features."""

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

    def read_data(self):
        """Extracts 278 features from the specified file for each profile."""

        with open(self.file_path) as json_file:
            dictionary = json.load(json_file)

        # Extract the data file name (we assume name to be between last \ and .)
        path = self.file_path
        dot = path.rfind('.')
        last_slash = path.rfind('\\')
        name = path[last_slash+1:dot]

        # Read flattened file
        raw_df = pd.DataFrame.from_dict(dictionary, orient='index')
        raw_df.reset_index(level=0, inplace=True)
        json_struct = json.loads(raw_df.to_json(orient="records"))  
        flat_df = pd.json_normalize(json_struct) 

        # Change "Locus.TPOX.SumAF_TPOX" to "SumAF_TPOX" to match features
        new_column_names = []
        for column_name in flat_df.columns: 
            if 'Locus' in column_name:
                new_column_name = column_name[column_name.rfind(".")+1:]
            else:
                new_column_name = column_name
            new_column_names.append(new_column_name) 
        flat_df.columns = new_column_names

        # Filter features to only include the ones specified in features
        data = flat_df    
        data.set_index('index', inplace=True)
        data.index.name = None
        data = data.astype(float)

        # # Translate features to more understandable strings.
        # with open('feature_translations.json') as json_file:
        #     translations = json.load(json_file)
        # data.rename(columns = translations, inplace = True)

        # Grab the outcome- and feature names.
        outcome_name = 'NOC'
        feature_names = data.columns[~data.columns.isin([outcome_name])].values.tolist()

        # Create the dictionary object to pass to data.
        params = { 'data' : data, 
                   'name' : name,
                   'feature_names' : feature_names, 
                   'outcome_name' : outcome_name, 
                   'test_size' : self.test_size }

        return params
