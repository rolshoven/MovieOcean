import os
import pickle
import pandas as pd
import numpy as np
from utils import cross_validate
from tqdm import tqdm
import matplotlib.pyplot as plt

NEIGHBORHOOD_SIZE = 15
SIMILARITIES = {'rating': 'cosine', 'personality': 'euclidean'}
STRATEGIES = ['rating', 'personality']
METRICS = ['rmse', 'f1']
CROSS_VALIDATION_SIZE = 3
PERSONALITY_FACTOR = 50
DEMOGRAPHIC_FACTORS = list(range(11))


def save_to_disk(filename, dictionary):
    with open(os.path.join('data', filename), 'wb') as file:
        pickle.dump(dictionary, file)


def plot(data, metric):
    assert metric in METRICS
    mets = ['RMSE', 'F1-Score']
    title = f'Influence of the Demographic Factor on the {mets[METRICS.index(metric)]}'
    scores_personality = [data[d]['personality'][0 if metric == 'rmse' else 1] for d in DEMOGRAPHIC_FACTORS]
    scores_rating = [data[d]['rating'][0 if metric == 'rmse' else 1] for d in DEMOGRAPHIC_FACTORS]
    plt.plot(DEMOGRAPHIC_FACTORS, scores_personality, label='Personality-based', color='cornflowerblue')
    plt.plot(DEMOGRAPHIC_FACTORS, scores_rating, label='Rating-based', color='burlywood')
    plt.xticks(DEMOGRAPHIC_FACTORS)
    plt.xlabel('Demographic Factor')
    plt.ylabel(mets[METRICS.index(metric)])
    plt.legend()
    plt.title(title)
    plt.savefig(f'plots/demographic_factor/demographic_factor_{metric}.jpg')
    plt.clf()


ratings = pd.read_csv(os.path.join('data', 'ratings.csv'))
personalities = pd.read_csv(os.path.join('data', 'personalities.csv'))
demographics = pd.read_csv(os.path.join('data', 'demographics.csv'))

users = ratings.UserId.unique()

# Dictionary that stores a list of tuples of RMSE and F1-Scores for each demographic factor
all_scores = {d: {s: [] for s in STRATEGIES} for d in DEMOGRAPHIC_FACTORS}

# Stores the computed metric tuple means (rmse, f1) for each demographic factor
mean_scores = {d: {s: None for s in STRATEGIES} for d in DEMOGRAPHIC_FACTORS}

for user in tqdm(users):
    for approach in STRATEGIES:
        for d in DEMOGRAPHIC_FACTORS:
            all_scores[d][approach].append(
                cross_validate(
                    data=ratings,
                    demographics=demographics,
                    user_id=user,
                    num_vals=CROSS_VALIDATION_SIZE,
                    num_neighbors=NEIGHBORHOOD_SIZE,
                    strategy=approach,
                    demographic_factor=d,
                    sim_metric=SIMILARITIES[approach],
                    personalities=personalities,
                    personality_factor=PERSONALITY_FACTOR
                )
            )

# Compute mean scores
for d in DEMOGRAPHIC_FACTORS:
    for approach in STRATEGIES:
        scores = all_scores[d][approach]
        mean_rmse = np.nanmean([s[0] for s in scores])
        mean_f1 = np.nanmean([s[1] for s in scores])
        mean_scores[d][approach] = (mean_rmse, mean_f1)

save_to_disk(os.path.join('demographic_factor', 'demographic_factor_all_scores.pkl'), all_scores)
save_to_disk(os.path.join('demographic_factor', 'demographic_factor_mean_scores.pkl'), mean_scores)

# Print values
for s in STRATEGIES:
    print('Strategy:', s)
    for d in DEMOGRAPHIC_FACTORS:
        rmse, f1 = mean_scores[d][s]
        print('\tDemographic Factor:', d)
        print('\t\t> RMSE:', format(rmse, '.3f'))
        print('\t\t> F1-Score:', format(f1, '.3f'))


# Plot scores
for m in METRICS:
    plot(mean_scores, m)
