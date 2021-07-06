import glob
import json
import pandas as pd

############################# Merge json objects from files #############################

# filenames = [r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features1_5000.txt',
#             r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features2_5000.txt',
#             r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features3_5000.txt',
#             r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features4_5000.txt',
#             r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5_5000.txt']

# result = []
# for f in filenames:
#     with open(f, 'r') as infile:
#         result.append(json.load(infile))

# dict_result = d = {**result[0], **result[1], **result[2], **result[3], **result[4]}

# with open(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_278.txt', 'w') as outfile:
#      json.dump(dict_result, outfile)

############################# Merge dataframes from csv files #############################

file1 = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\features5000\Features5000_19.txt'
file2 = r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features590_19.txt'

data1 = pd.read_csv(file1, index_col=0)
data2 = pd.read_csv(file2, index_col=0)

data_merged = data1.append(data2)
data_merged.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\data\Features_merged_19.txt')