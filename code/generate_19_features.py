import json
import pandas as pd

file_path =  r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_278.txt'# Full path to 276 features file.

# The sample name (index) + 19 features + NOC
features = ['index','MAC', 'TAC', 'MinNOC_CSF1PO', 'MinNOC_D16S539', 
            'PercAF_D1S1656', 'AlleleCount_D3S1358', 
            'AlleleCount_D8S1179', 'MinNOC_Penta D', 'MinNOC_Penta E', 
            'SumAF_TH01', 'AlleleCount_TPOX', 'MinNOC_TPOX', 
            'stdHeight_vWA', 'stdAllele', 'MAC0', 'MAC5-6', 
            'peaksBelowRFU', 'MatchProbability', 'MinNOC', 'NOC']

with open(file_path) as json_file:
    dictionary = json.load(json_file)

# Read flattened file
raw_df = pd.DataFrame.from_dict(dictionary, orient='index')
raw_df.reset_index(level=0, inplace=True)
json_struct = json.loads(raw_df.to_json(orient='records'))  
flat_df = pd.json_normalize(json_struct) 

# Change "Locus.TPOX.SumAF_TPOX" to "SumAF_TPOX" to match features
new_column_names = []
for column_name in flat_df.columns: 
    if 'Locus' in column_name:
        new_column_name = column_name[column_name.rfind('.')+1:]
    else:
        new_column_name = column_name
    new_column_names.append(new_column_name) 
flat_df.columns = new_column_names

# Filter features to only include the ones specified in features
data = flat_df[features]    
data.set_index('index', inplace=True)
data.index.name = None
data = data.astype(float)

# Put in csv
data.to_csv('new_features')# Full path to 19 features output file.