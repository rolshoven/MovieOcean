import os
import pickle
import pandas as pd
import numpy as np
from utils import cross_validate
from tqdm import tqdm
import matplotlib.pyplot as plt

NEIGHBORHOOD_SIZES = list(range(5, 16, 2))
SIMILARITIES = ['cosine', 'jaccard', 'euclidean', 'pearson']
STRATEGIES = ['rating', 'personality']
METRICS = ['rmse', 'f1']
CROSS_VALIDATION_SIZE = 3
PERSONALITY_FACTOR = 1
DEMOGRAPHIC_FACTOR = 1


def save_to_disk(filename, dictionary):
    with open(os.path.join('data', filename), 'wb') as file:
        pickle.dump(dictionary, file)


def plot(data, sim, metric):
    assert metric in METRICS
    sims = ['Cosine Similarity', 'Jaccard Index', 'Inverse Euclidean Distance',
            'Absolute Pearson Corellation Coefficient']
    mets = ['RMSE', 'F1-Score']
    title = f'Comparison of Personality-based and Rating-Based {mets[METRICS.index(metric)]}\n' \
            f'When Using {sims[SIMILARITIES.index(sim)]}'
    scores_personality = [data[sim][n]['personality'][0 if metric == 'rmse' else 1] for n in NEIGHBORHOOD_SIZES]
    scores_rating = [data[sim][n]['rating'][0 if metric == 'rmse' else 1] for n in NEIGHBORHOOD_SIZES]
    plt.plot(NEIGHBORHOOD_SIZES, scores_personality, label='Personality-based', color='cornflowerblue')
    plt.plot(NEIGHBORHOOD_SIZES, scores_rating, label='Rating-based', color='burlywood')
    plt.xticks(NEIGHBORHOOD_SIZES)
    plt.xlabel('Neighborhood size')
    plt.ylabel(mets[METRICS.index(metric)])
    plt.title(title)
    plt.legend()
    plt.savefig(f'plots/grid_search/{sim}_{metric}.jpg')
    plt.clf()


ratings = pd.read_csv(os.path.join('data', 'ratings.csv'))
personalities = pd.read_csv(os.path.join('data', 'personalities.csv'))
demographics = pd.read_csv(os.path.join('data', 'demographics.csv'))

users = ratings.UserId.unique()

# Dictionary of dictionaries of dictionaties (first similarity metric, then neighborhood size, then strategy)
# Stores the computed metric tuples (rmse, f1) for each combination and each user.
all_scores = {sim: {num_neighbors: {s: [] for s in STRATEGIES} for num_neighbors in NEIGHBORHOOD_SIZES} for sim in
              SIMILARITIES}

# Stores the computed metric tuple means (rmse, f1) for each combination.
mean_scores = {sim: {num_neighbors: {s: None for s in STRATEGIES} for num_neighbors in NEIGHBORHOOD_SIZES} for sim in
               SIMILARITIES}

for user in tqdm(users):
    for sim in SIMILARITIES:
        for num_neighbors in NEIGHBORHOOD_SIZES:
            for approach in STRATEGIES:
                all_scores[sim][num_neighbors][approach].append(
                    cross_validate(
                        data=ratings,
                        demographics=demographics,
                        user_id=user,
                        num_vals=CROSS_VALIDATION_SIZE,
                        num_neighbors=num_neighbors,
                        strategy=approach,
                        demographic_factor=DEMOGRAPHIC_FACTOR,
                        sim_metric=sim,
                        personalities=personalities,
                        personality_factor=1
                    )
                )

# Compute mean scores
for sim in SIMILARITIES:
    for num_neighbors in NEIGHBORHOOD_SIZES:
        for approach in STRATEGIES:
            scores = all_scores[sim][num_neighbors][approach]
            mean_rmse = np.nanmean([s[0] for s in scores])
            mean_f1 = np.nanmean([s[1] for s in scores])
            mean_scores[sim][num_neighbors][approach] = (mean_rmse, mean_f1)

save_to_disk(os.path.join('grid_search', 'grid_search_all_scores.pkl'), all_scores)
save_to_disk(os.path.join('grid_search', 'grid_search_mean_scores.pkl'), mean_scores)

# Print values
for sim in mean_scores.keys():
    print('Similarity Measure:', sim)
    for size in mean_scores[sim].keys():
        print('\tNeighborhood Size:', size)
        for approach in mean_scores[sim][size].keys():
            rmse, f1 = mean_scores[sim][size][approach]
            print('\t\tStrategy:', approach)
            print('\t\t\t> RMSE:', format(rmse, '.3f'))
            print('\t\t\t> F1-Score:', format(f1, '.3f'))

# Plot scores
for sim in SIMILARITIES:
    for m in METRICS:
        plot(mean_scores, sim, m)
