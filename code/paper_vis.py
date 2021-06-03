import numpy as np
import pandas as pd
import math
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# options_1 = ['stayed 2, 3, 4', 'stayed 3', 'stayed 3, 4', '3, 4 to 2, 3, 4', '2, 3, 4 to 3, 4']
# options_2 = ['stayed i don\'t know', 'stayed yes', 'yes to i don\'t know', 'i don\'t know to yes']
# options_3 = ['3, 4, 5 to 4', '3, 4 to 3', '3, 4, 5 to 3, 4', '2, 3, 4, 5 to 3, 4', 'stayed 3, 4, 5', 'stayed 3, 4']

# pred_to_shap_1 = [1, 1, 1, 1, 3]
# pred_to_vis_1 = [0, 1, 2, 0, 4]

# pred_to_cf_2 = [1, 0, 0, 1, 1, 4]
# pred_to_vis_2 = [0, 1, 1, 1, 1, 3]

# pred_1 = [4, 1, 2, 0, 0]
# shap_1 = [2, 1, 4, 0, 0]
# vis_1 = [0, 1, 6, 0, 0]
# purple = '#673ab7'

# fig, axs = plt.subplots(1, 2, sharey=True, figsize=(17,3), tight_layout=True)

# axs[0].bar(options_3, pred_to_cf_2, color=purple)
# axs[1].bar(options_3, pred_to_vis_2, color=purple)

# axs[0].set_title('Changes to considered NOC from only the prediction of 4 (3.53) to the CF table explanation')
# axs[1].set_title('Changes to considered NOC from only the prediction of 4 (3.53) to the compound visualization')

# plt.show(block=True)

purples = ['#522f8f', '#937dba', '#d5caeb']

fig, axs = plt.subplots(2, 1, sharey=True, figsize=(12, 8), tight_layout=True)
labels_profile_1 = ['Correctness (SHAP)', 'Correctness (visualization)', 'Pinpoint NOC (SHAP)', 'Pinpoint NOC (visualization)']
labels_profile_2 = ['Correctness (CF table)', 'Correctness (visualization)', 'Pinpoint NOC (CF table)', 'Pinpoint NOC (visualization)']

less_trust_profile_1 = np.array([1, 0, 1, 0])
same_trust_profile_1 = np.array([3, 4, 3, 3])
more_trust_profile_1 = np.array([3, 3, 3, 4])

less_trust_profile_2 = np.array([0, 2, 0, 0])
same_trust_profile_2 = np.array([7, 5, 5, 4])
more_trust_profile_2 = np.array([0, 0, 2, 3])

axs[0].bar(labels_profile_1, less_trust_profile_1, label='Less trust', color=purples[2])
axs[0].bar(labels_profile_1, same_trust_profile_1, bottom=less_trust_profile_1, label='Same trust', color=purples[1])
axs[0].bar(labels_profile_1, more_trust_profile_1, bottom=(less_trust_profile_1 + same_trust_profile_1), label='More trust', color=purples[0])
title = (('Profile 1: Change of users\' trust in the correctness of the prediction after seeing the explanation \n') +
('and change of users\' trust to pinpoint the NOC after seeing the explanation'))
axs[0].set_title(title, loc='left')


axs[1].bar(labels_profile_2, less_trust_profile_2, label='Less trust', color=purples[2])
axs[1].bar(labels_profile_2, same_trust_profile_2, bottom=less_trust_profile_2, label='Same trust', color=purples[1])
axs[1].bar(labels_profile_2, more_trust_profile_2, bottom=(less_trust_profile_2 + same_trust_profile_2), label='More trust', color=purples[0])
title = (('Profile 2: Do users have more trust in the correctness of the prediction after seeing the explanation \n') +
        ('and do users have more trust to pinpoint the NOC after seeing an explanation'))
axs[1].set_title(title, loc='left')

handles, labels = axs[0].get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
#fig.legend(*zip(*unique))

plt.show(block=True)