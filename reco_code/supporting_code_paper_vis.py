import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


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