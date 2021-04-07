
### CAN BE USED IN MAIN TO GENERATE THE DIST SCORE AND FEATURES CHANGED SCORE ###
### WILL NOT COMPILE ON ITS OWN ###

i = 0

for index, dp in dataset_merged.train_data.iterrows():
    if i < 2005:
        i+=1
        continue
    print(i)
    dp_scaled = dataset_merged.scale_data_point(dp)
    dp_pred = rf_regressor_merged.get_prediction(dp[dataset_merged.feature_names])
    cf_target = (rf_regressor_merged.get_secondary_prediction(dp[dataset_merged.feature_names]))
    scores = CF_generator.generate_weighted_counterfactual(dp, dp_scaled, cf_target)
    scores.to_csv(r'D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\weights\{}'.format(i) + '.csv')

    i+=1

import glob
combined_csv = pd.concat( [ pd.read_csv(f) for f in glob.glob(r"D:\Documenten\TUdelft\thesis\mep_veldhuis\code\evaluation\weights\*") ] )
combined_csv.to_csv( "all.csv", index=False )


eval = pd.read_csv('all.csv', index_col=0)

plt.hist(eval['dist'], color='tab:orange', alpha = 0.5, label='Distance')
plt.hist(5*eval['dist'], color='tab:olive', alpha = 0.5, label='w * Distance')
plt.hist(eval['features'], color='tab:blue', alpha = 0.5, label='Features changed')
plt.gca().set(title='Distribution of scores on training set', ylabel='Score')
plt.legend()
plt.show()

print('Distance mean: {}'.format(np.mean(eval['dist'])))
print('Distance median: {}'.format(np.median(eval['dist'])))
print('Distance minimum: {}'.format(np.min(eval['dist'])))
print('Distance maximum: {}'.format(np.max(eval['dist'])))
print('Features changed mean: {}'.format(np.mean(eval['features'])))
print('Features changed median: {}'.format(np.median(eval['features'])))
print('Features changed minimum: {}'.format(np.min(eval['features'])))
print('Features changed maximum: {}'.format(np.max(eval['features'])))