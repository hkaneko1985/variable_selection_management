# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

similarity_index = 'corr'
# 'corr': correlation coefficient
# 'mic': Maximal Information Coefficient (MIC) [please install minepy https://minepy.readthedocs.io/en/latest/]
# 'rbf': Gaussian kernel

import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
dataset = pd.read_csv('sample_dataset.csv', index_col=0)
selected_variable_numbers = pd.read_csv('sample_selected_variable_numbers.csv', index_col=0)
selected_variable_numbers = list(selected_variable_numbers.iloc[:, 0])
removed_variable_numbers = list(set(range(dataset.shape[1])) - set(selected_variable_numbers))

# similarity
if similarity_index == 'corr':
    similarity_matrix = abs(dataset.corr())
    similarity_matrix = similarity_matrix.iloc[selected_variable_numbers, removed_variable_numbers]
elif similarity_index == 'mic':
    from minepy import MINE
    similarity_matrix = pd.DataFrame(columns=dataset.columns[removed_variable_numbers], index=dataset.columns[selected_variable_numbers])
    mine = MINE(alpha=0.6, c=15, est='mic_approx')
    for column in similarity_matrix.columns:
        for index in similarity_matrix.index:
            mine.compute_score(dataset[column], dataset[index])
            similarity_matrix.loc[index, column] = mine.mic()
    similarity_matrix = similarity_matrix.astype(float)
elif similarity_index == 'rbf':
    from scipy.spatial.distance import cdist
    gamma = 1 / dataset.shape[1]
    autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()
    similarity_matrix = np.exp(-gamma * cdist(autoscaled_dataset.iloc[:, selected_variable_numbers].T, autoscaled_dataset.iloc[:, removed_variable_numbers].T, metric='sqeuclidean'))
    similarity_matrix = pd.DataFrame(similarity_matrix, columns=dataset.columns[removed_variable_numbers], index=dataset.columns[selected_variable_numbers])

# heat map
plt.rcParams['font.size'] = 12
sns.heatmap(similarity_matrix, vmax=1, vmin=0, cmap='seismic', xticklabels=1, yticklabels=1)
plt.xlim([0, similarity_matrix.shape[1]])
plt.ylim([0, similarity_matrix.shape[0]])
plt.show()

# save
similarity_matrix.to_csv('similarity_matrix_{0}.csv'.format(similarity_index))
