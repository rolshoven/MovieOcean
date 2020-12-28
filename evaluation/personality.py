import os
import pickle
import pandas as pd
import numpy as np
from utils import cross_validate
from tqdm import tqdm
import matplotlib.pyplot as plt

NEIGHBORHOOD_SIZE = 15
SIMILARITIY = 'euclidean'
STRATEGY = 'personality'
METRICS = ['rmse', 'f1']
CROSS_VALIDATION_SIZE = 3
PERSONALITY_FACTORS = list(range(0, 51, 5))
DEMOGRAPHIC_FACTOR = 1


def save_to_disk(filename, dictionary):
    with open(os.path.join('data', filename), 'wb') as file:
        pickle.dump(dictionary, file)


def plot(data, metric):
    assert metric in METRICS
    mets = ['RMSE', 'F1-Score']
    title = f'Influence of the Personality Factor on the {mets[METRICS.index(metric)]}'
    scores = [data[p][0 if metric == 'rmse' else 1] for p in PERSONALITY_FACTORS]
    plt.plot(PERSONALITY_FACTORS, scores, color='cornflowerblue')
    plt.xticks(PERSONALITY_FACTORS)
    plt.xlabel('Personality Factor')
    plt.ylabel(mets[METRICS.index(metric)])
    plt.title(title)
    plt.savefig(f'plots/personality_factor/personality_factor_{metric}.jpg')
    plt.clf()



ratings = pd.read_csv(os.path.join('data', 'ratings.csv'))
personalities = pd.read_csv(os.path.join('data', 'personalities.csv'))
demographics = pd.read_csv(os.path.join('data', 'demographics.csv'))

users = ratings.UserId.unique()

# Dictionary that stores a list of tuples of RMSE and F1-Scores for each personality factor
all_scores = {p: [] for p in PERSONALITY_FACTORS}

# Stores the computed metric tuple means (rmse, f1) for each personality factor
mean_scores = {p: None for p in PERSONALITY_FACTORS}

for user in tqdm(users):
    for p in PERSONALITY_FACTORS:
        all_scores[p].append(
            cross_validate(
                data=ratings,
                demographics=demographics,
                user_id=user,
                num_vals=CROSS_VALIDATION_SIZE,
                num_neighbors=NEIGHBORHOOD_SIZE,
                strategy=STRATEGY,
                demographic_factor=DEMOGRAPHIC_FACTOR,
                sim_metric=SIMILARITIY,
                personalities=personalities,
                personality_factor=p
            )
        )

# Compute mean scores
for p in PERSONALITY_FACTORS:
    mean_rmse = np.nanmean([s[0] for s in all_scores[p]])
    mean_f1 = np.nanmean([s[1] for s in all_scores[p]])
    mean_scores[p] = (mean_rmse, mean_f1)


save_to_disk(os.path.join('personality_factor', 'personality_factor_all_scores.pkl'), all_scores)
save_to_disk(os.path.join('personality_factor', 'personality_factor_mean_scores.pkl'), mean_scores)

# Print values
for p in PERSONALITY_FACTORS:
    print('Personality Factor', p)
    print('\t> RMSE:', format(mean_scores[p][0], '.3f'))
    print('\t> F1-Score:', format(mean_scores[p][1], '.3f'))

# Plot scores
for m in METRICS:
    plot(mean_scores, m)
