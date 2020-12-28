import os
import pandas as pd
import numpy as np
import warnings


def get_common_ratings(data, uid1, uid2):
    """Returns a list of movies that both users rated and their ratings as numpy arrays.

    :param data: a pandas DataFrame with columns UserId, MovieId and Rating
    :param uid1: the (randomized) id of the first user
    :param uid2: the (randomized) id of the second user
    :return: a tuple (movie_ids, u1_ratings, u2_ratings) containing the movie ids and the ratings
             of each user for the movies as a numpy array
    """
    u1_df = data[data.UserId == uid1]
    u2_df = data[data.UserId == uid2]
    rated_by_both = u1_df.merge(u2_df, on='MovieId')
    movie_ids = list(rated_by_both.MovieId)
    u1_ratings = rated_by_both.Rating_x.values
    u2_ratings = rated_by_both.Rating_y.values
    return movie_ids, u1_ratings, u2_ratings


def get_rating_distances(data):
    """Returns a DataFrame containing the distances between all users.

    :param data: a Pandas DataFrame with columns UserId, MovieId and Rating
    :return: a Pandas DataFrame with columns UserId1, UserId2, Distance
    """
    rating_distance_data = []
    user_ids = data.UserId.unique()
    for i in range(len(user_ids) - 1):
        for j in range(i + 1, len(user_ids)):
            u1 = user_ids[i]
            u2 = user_ids[j]
            _, r1, r2 = get_common_ratings(data, u1, u2)
            if len(r1) > 0:
                rating_distance_data.append([u1, u2, np.linalg.norm(r1 - r2)])
            else:
                rating_distance_data.append([u1, u2, np.inf])
    return pd.DataFrame(rating_distance_data, columns=['UserId1', 'UserId2', 'Distance'])


def get_mean_rating(data, uid):
    """Returns the mean rating of a user within a given DataFrame.

    :param data: the DataFrame with columns UserId, MovieId and Rating
    :param uid: the id of the user
    :return: the mean rating of the user
    """
    ratings = data[data.UserId == uid].Rating.values
    return ratings.mean() if len(ratings) > 0 else np.nan


# def get_neighborhood(data, user_id, max_neighbors):
#     """Returns a list of user ids that are in the neighborhood of a specified user.
#
#     :param data: a DataFrame with columns UserId1, UserId2 and Distance
#     :param user_id: the id of the user whose neighborhood will be computed
#     :param max_neighbors: the maximum number of neighbors
#     :return: a list of user ids that correspond to the neighborhood of the user.
#     """
#     checked = []
#     # Prepare lists of user distances and user ids
#     neighbor_ids = [-1 for _ in range(max_neighbors)]
#     neighbor_distances = [np.inf for _ in range(max_neighbors)]
#
#     # Concatenate entries where UserId1 or UserId2 corresponds to user_id
#     df1 = data[data.UserId1 == user_id]
#     df2 = data[data.UserId2 == user_id]
#     df_user = pd.concat([df1, df2])
#
#     # Search for the minimum distances
#     for idx in range(len(df_user)):
#         row = df_user.iloc[idx]
#         users = set([row.UserId1, row.UserId2])
#         if users not in checked:
#             checked.append(users)
#             if max(neighbor_distances) > row.Distance:
#                 i = np.argmax(neighbor_distances)
#                 neighbor_distances[i] = row.Distance
#                 neighbor_ids[i] = row.UserId1 if row.UserId1 != user_id else row.UserId2
#
#     # If there are less than max_neighbor neighbors, remove them
#     neighbor_ids = [int(i) for i in neighbor_ids if i != -1]
#
#     return neighbor_ids

def get_rating_neighborhood(data, user_id, max_neighbors, sim_metric='cosine'):
    """Returns a list of user ids that are in the neighborhood of a specified user.

    :param data: a DataFrame with columns UserId, MovieId, Rating
    :param user_id: the id of the user whose neighborhood will be computed
    :param max_neighbors: the maximum number of neighbors
    :param sim_metric: The similarity measure that is used to compute the neighborhoods.
                       Can be either cosine, jaccard, euclidean or pearson, which correspond
                       to the Cosine similarity, Jaccard similarity, inverted Euclidean
                       distance of the Pearson correlation coefficient.
    :return: a list of user ids that correspond to the neighborhood of the user.
    """
    assert sim_metric in ['cosine', 'jaccard', 'euclidean', 'pearson'], 'Wrong similarity metric specifier used!'

    if sim_metric == 'cosine':
        sim_func = cosine_similarity
    elif sim_metric == 'jaccard':
        sim_func = jaccard_rating_similarity
    elif sim_metric == 'euclidean':
        sim_func = inverted_euclidean_distance
    else:
        sim_func = pearson_correlation_coefficient

    users = list(data[data.UserId != user_id].UserId.unique())
    similarities = []

    # Get ratings of the active user
    au_ratings = get_ratings(data, [user_id])[0]

    # Compute similarities
    for user in users:
        ratings = get_ratings(data, [user])[0]
        similarities.append(sim_func(au_ratings, ratings))

    # Filter the most similar users
    num_neighbors = max_neighbors if max_neighbors < len(users) else len(users)
    most_similar_indices = np.argpartition(similarities, -num_neighbors)[-num_neighbors:]
    neighbor_ids = [uid for idx, uid in enumerate(users) if idx in most_similar_indices]

    return neighbor_ids


def get_personality_neighborhood(data, user_id, max_neighbors, sim_metric='cosine'):
    """Returns a list of user ids that are in the neighborhood of a specified user.

    :param data: a DataFrame with columns UserId, Extraversion, Agreeableness, Conscientiousness,
                 Neuroticism and Openness
    :param user_id: the id of the user whose neighborhood will be computed
    :param max_neighbors: the maximum number of neighbors
    :param sim_metric: The similarity measure that is used to compute the neighborhoods.
                       Can be either cosine, euclidean or pearson, which correspond
                       to the Cosine similarity, inverted Euclidean
                       distance and the Pearson correlation coefficient.
    :return: a list of user ids that correspond to the neighborhood of the user.
    """
    assert sim_metric in ['cosine', 'euclidean', 'pearson'], 'Wrong similarity metric specifier used!'

    if sim_metric == 'cosine':
        sim_func = cosine_similarity
    elif sim_metric == 'euclidean':
        sim_func = inverted_euclidean_distance
    else:
        # Don't use significance weighting for building the neighborhoods
        # It shouldn't make a difference but it is useless
        def pcc_wrapper(v1, v2):
            return pearson_correlation_coefficient(v1, v2, significance_weighting=False)
        sim_func = pcc_wrapper

    users = list(data[data.UserId != user_id].UserId.unique())
    similarities = []

    # Get ratings of the active user
    au_personality = get_personalities(data, [user_id])[0]

    # Compute similarities
    for user in users:
        personalities = get_personalities(data, [user])[0]
        similarities.append(sim_func(au_personality, personalities))

    # Filter the most similar users
    num_neighbors = max_neighbors if max_neighbors < len(users) else len(users)
    most_similar_indices = np.argpartition(similarities, -num_neighbors)[-num_neighbors:]
    neighbor_ids = [uid for idx, uid in enumerate(users) if idx in most_similar_indices]

    return neighbor_ids


def get_rating(data, user_id, movie_id):
    """Returns the rating of a certain movie given by a certain user if it exists.

    :param data: a DataFrame with columns UserId, MovieId and Rating
    :param user_id: the id of the user whose rating should be fetched
    :param movie_id: the id of the movie for which we want to fetch the rating
    :return: the rating of the user for this specific movie if it exists, otherwise None.
    """
    user_ratings = data[data.UserId == user_id]
    rating = user_ratings[user_ratings.MovieId == movie_id]
    if len(rating) > 0:
        return rating.Rating.values[0]
    return None


def get_ratings(data, users):
    """Returns a list of rating vectors for the specified users.

    :param data: a DataFrame with columns UserId, MovieId and Rating
    :param users: a list of users for which the rating vectors are computed
    :return: a list of rating vectors that correspond to the rating of the specified users.
    """
    movies = list(data.MovieId.unique())
    ratings = np.zeros((len(users), len(movies)))
    for row, uid in enumerate(users):
        rating_data = data[data.UserId == uid].iloc[:, 1:3]
        for i in range(len(rating_data)):
            mid, score = rating_data.iloc[i]
            ratings[row, movies.index(mid)] = score
    return list(ratings)


def get_personalities(data, users):
    """Returns a list of personality vectors for the specified users.

    :param data: a DataFrame with columns UserId, Extraversion, Agreeableness, Conscientiousness,
                 Neuroticism and Openness
    :param users: a list of users for which the personality vectors are returned
    :return: a list of vectors that correspond to the Big Five personalities of the specified users.
    """
    # Keep track of personalities
    personalities = []

    # Save Big Five scores in OCEAN order
    ocean_indices = [4, 2, 0, 1, 3]

    for uid in users:
        ocean_scores = data[data.UserId == uid].iloc[0, 1:].iloc[ocean_indices].values
        personalities.append(ocean_scores)

    return personalities


def cosine_similarity(v1, v2):
    """Returns the cosine similarity between two vectors.

    :param v1: the first vector
    :param v2: the second vector
    :return: the cosine similarity between the two vectors
    """
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return v1.dot(v2) / denom if denom > 0 else 0


def jaccard_rating_similarity(v1, v2):
    """Computes the Jaccard similarity between two rating vectors v1 and v2.

    :param v1: the first rating vector, containing the ratings of the movies or 0 if a movie hasn't been rated.
    :param v2: the second rating vector, containing the ratings of the movies or 0 if a movie hasn't been rated.
    :return: the Jaccard similarity between the two rating vectors.
    """
    # Discard movies that were not rated by either of the users
    rated_at_least_once = (v1 + v2) != 0
    v1 = v1[rated_at_least_once]
    v2 = v2[rated_at_least_once]

    # Define the numerator and the denominator for computing the Jaccard similarity
    rated_by_both = np.sum((v1 * v2) != 0)
    total_rated_movies = len(v1)

    return rated_by_both / total_rated_movies if total_rated_movies > 0 else 0


def pearson_correlation_coefficient(v1, v2, significance_weighting=True, gamma=25):
    """Returns the the Pearson correlation coefficient between two vectors.

    :param v1: the first vector
    :param v2: the second vector
    :param significance_weighting: whether to apply significance weighting or not
    :param gamma: the significance weight parameter (default is 25)
    :return: the Pearson correlation coefficient between the two vectors
    """
    # If one vector is entirely zero, there can be no Pearson correlation between the vectors
    if (v1 == 0).all() or (v2 == 0).all():
        return 0

    # Get mean rating of users (must happen before we modify v1 and v2)
    m1 = np.mean(v1[v1 > 0])
    m2 = np.mean(v2[v2 > 0])

    # Discard movies that were not rated by both users (has no effect on personalities since you cannot score 0 points)
    rated_by_both = (v1 > 0) * (v2 > 0)
    v1 = v1[rated_by_both].astype(np.float64)
    v2 = v2[rated_by_both].astype(np.float64)

    # If they have no ratings in common, there is no correlation (again, no effect on personalities)
    if len(v1) == 0:
        return 0

    # Apply mean centering
    v1 -= m1
    v2 -= m2

    # If one variable/vector has zero variance, the correlation is not defined
    if (v1 == 0).all() or (v2 == 0).all():
        return np.nan

    pcc = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    if significance_weighting:
        # Significance weight
        w = min(len(v1), gamma) / gamma
        pcc *= w

    return pcc


def inverted_euclidean_distance(v1, v2):
    """Returns the normalized inverted Euclidean distance between two vectors, which can serve as a similarity measure.

    :param v1: the first vector
    :param v2: the second vector
    :return: a similarity value between 0 and 1 based on the inverted Euclidean distance.
    """
    return 1 / (np.linalg.norm(v1 - v2) + 1)


def demographic_similarity(data, uid1, uid2, attribute_weights=(1, 1, 1)):
    """Returns the demographic similarity between two users.

    :param data: a DataFrame with columns UserId, CountryId, GenderId and Age
    :param uid1: the id of the first user
    :param uid2: the id of the second user
    :param attribute_weights: a triplet of weights corresponding to the importance of the country, gender and age.
                              The default value is (1, 1, 1) which gives equal importance to all attributes.
    :return: the demographic similarity between the two users
    """
    # Get user data
    d1 = data[data.UserId == uid1]
    d2 = data[data.UserId == uid2]

    # Compute similarity between the attributes
    country_sim = int(d1.CountryId.values == d2.CountryId.values)
    gender_sim = int(d1.GenderId.values == d2.GenderId.values)
    age_sim = 1 / (np.abs(d1.Age.values - d2.Age.values) + 1)[0]

    # Create a vector of similarities and another one with the attribute weights
    dvec = np.array([country_sim, gender_sim, age_sim])
    weights = np.array(attribute_weights)

    # Compute the demographic similarity
    dsim = dvec.dot(weights) / np.sum(weights)

    return dsim


def predict_rating_based(data, demographics, movies, user_id, num_neighbors, sim_metric, demographic_factor):
    """Returns the predicted ratings on given items using the standard Collaborative Filtering algorithm.

    This function makes use of the standard Collaborative Filtering algorithm to compute the predicted
    ratings for each movie in the list movies for the user with the specified user id. The algorithm
    will build neighborhoods that consist of at most num_neighbors users. Note that sometimes it is not
    possible to predict a rating for a certain movie because no one in the neighborhood rated it. In this
    case, the list that is returned will contain np.nan values, you should keep that in mind.

    :param sim_metric:
    :param data: the rating DataFrame with columns UserId, MovieId and Rating
    :param demographics: the demographics DataFrame with columns UserId, CountryId, GenderId and Age
    :param movies: a list of movie ids for which the ratings are predicted
    :param user_id: the id of the active user
    :param num_neighbors: the number of neighbors that are considered in the algorithm
    :param sim_metric: The similarity measure that is used to compute the neighborhoods and to serve
                       as a weight in the CF algorithm. Can be either cosine, jaccard, euclidean or pearson,
                       which correspond to the Cosine similarity, Jaccard similarity, inverted Euclidean
                       distance of the Pearson correlation coefficient.
    :param demographic_factor: the importance of the demographic similarity compared to the rating similarity.
    :return: a list of predicted ratings for each movie in the list movies
    """
    assert sim_metric in ['cosine', 'jaccard', 'euclidean', 'pearson'], 'Wrong similarity metric specifier used!'

    if sim_metric == 'cosine':
        sim_func = cosine_similarity
    elif sim_metric == 'jaccard':
        sim_func = jaccard_rating_similarity
    elif sim_metric == 'euclidean':
        sim_func = inverted_euclidean_distance
    else:
        sim_func = pearson_correlation_coefficient

    # Compute the neighborhood
    neighbors = get_rating_neighborhood(data, user_id, num_neighbors, sim_metric=sim_metric)

    # For each movie that was held out, compute the predictions
    y_pred = []
    au_mr = get_mean_rating(data, user_id)
    for movie in movies:
        weighted_rating_deviations = 0
        similarity_normalization = 0
        for n in neighbors:
            r = get_rating(data, n, movie)
            if r:
                mr = get_mean_rating(data, n)
                r1, r2 = get_ratings(data, [user_id, n])
                rsim = sim_func(r1, r2)
                dsim = demographic_similarity(demographics, user_id, n)
                sim = (rsim + demographic_factor * dsim) / (1 + demographic_factor)
                weighted_rating_deviations += (r - mr) * sim
                similarity_normalization += np.abs(sim)
        if similarity_normalization > 0:
            y_pred.append(au_mr + weighted_rating_deviations / similarity_normalization)
        else:
            y_pred.append(np.nan)
    return y_pred


def predict_personality_based(data, personalities, demographics, movies, user_id, num_neighbors, personality_factor,
                              demographic_factor, sim_metric):
    """Returns the predicted ratings on given items using the the personality of the users and the CF algorithm.

    This function is a modified version of the Collaborative Filtering algorithm. It builds neighborhoods based on
    the similarity of the personality of the users. Afterwards, the CF algorithm with a modified similarity weight
    is used to predict the ratings. The modified similarity is a combination of the cosine rating similarity and a
    weighted personality similarity.  Note that sometimes it is not possible to predict a rating for a certain movie
    because no one in the neighborhood rated it. In this case, the list that is returned will contain np.nan values,
    you should keep that in mind.

    :param data: the rating DataFrame with columns UserId, MovieId and Rating
    :param personalities: the personality DataFrame with columns UserId, Extraversion, Agreeableness, Conscientiousness,
                          Neuroticism and Openness
    :param demographics: the demographics DataFrame with columns UserId, CountryId, GenderId and Age
    :param movies: a list of movie ids for which the ratings are predicted
    :param user_id: the id of the active user
    :param num_neighbors: the number of neighbors that are considered in the algorithm
    :param personality_factor: the importance of the personality in the similarity metric
    :param demographic_factor: the importance of the demographics in the similarity metric
    :param sim_metric: The similarity measure that is used to compute the neighborhoods and to serve
                       as a weight in the CF algorithm. Can be either cosine, jaccard, euclidean or pearson,
                       which correspond to the Cosine similarity, Jaccard similarity, inverted Euclidean
                       distance of the Pearson correlation coefficient.
    :return: a list of predicted ratings for each movie in the list movies
    """
    assert sim_metric in ['cosine', 'jaccard', 'euclidean', 'pearson'], 'Wrong similarity metric specifier used!'

    if sim_metric == 'cosine':
        sim_func = cosine_similarity
    elif sim_metric == 'jaccard':
        sim_func = jaccard_rating_similarity
    elif sim_metric == 'euclidean':
        sim_func = inverted_euclidean_distance
    else:
        sim_func = pearson_correlation_coefficient

    if sim_metric not in ['jaccard', 'pearson']:
        pers_sim_func = sim_func
    else:
        # Don't use jaccard similarity for personalities, we use the Pearson correlation instead
        # Also, if we were to use pcc in the first place, deactivate significance weighting for personalities
        def pcc_wrapper(v1, v2):
            return pearson_correlation_coefficient(v1, v2, significance_weighting=False)
        pers_sim_func = pcc_wrapper

    pers_sim_metric = sim_metric if sim_metric != 'jaccard' else 'pearson'

    # Compute the neighborhood with personality
    neighbors = get_personality_neighborhood(personalities, user_id, num_neighbors, sim_metric=pers_sim_metric)

    # For each movie that was held out, compute the predictions
    y_pred = []
    au_mr = get_mean_rating(data, user_id)  # mean rating of current user
    for movie in movies:
        # iterate through all neighbors
        num = 0
        denom = 0
        for n in neighbors:
            r = get_rating(data, n, movie)
            if r:
                # mean rating of user from neighborhood
                mr = get_mean_rating(data, n)

                # compute cosine similarity for ratings
                r1, r2 = get_ratings(data, [user_id, n])
                rsim = sim_func(r1, r2)

                # calculate similarity of personality
                p1, p2 = get_personalities(personalities, [user_id, n])
                psim = pers_sim_func(p1, p2)

                # calculate similarity of demographics
                dsim = demographic_similarity(demographics, user_id, n)

                # compute similarity with personality and rating
                sim = (rsim + personality_factor * psim + demographic_factor * dsim)
                sim /= (1 + personality_factor + demographic_factor)

                num += (r - mr) * sim
                denom += sim

        # Compute rating predictions using the collaborative filtering algorithm
        if denom > 0:
            y_pred.append(au_mr + num / denom)
        else:
            y_pred.append(np.nan)
    return y_pred


def compute_rmse(y_pred, y_true, ignore_warnings=True, print_summary=True):
    """Computes the root mean squared error between the predictions y_pred and the ground truth y_true.

    :param y_pred: a list of predicted ratings
    :param y_true: a list of real ratings
    :param ignore_warnings: ignores all warnings such as empty slices when a rating couldn't be predicted
    :param print_summary: print a summary of the predictions and ground truths together with the RMSE value
    :return: the RMSE for the given predictions and ground truths (can also be NaN)
    """

    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore', category=RuntimeWarning)
        y_pred = np.array(y_pred)
        rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))

    if print_summary:
        print('True ratings:\t\t', *[format(r, '05.2f') for r in y_true], sep='  ')
        print('Predicted ratings:\t', *[format(pr, '05.2f') if not np.isnan(pr) else '-----' for pr in y_pred],
              sep='  ')
        print('RMSE:', format(rmse, '.3f'))
        print()

    return rmse


# def get_recommendations(data, user_id, max_recommendations, num_neighbors, strategy, p_dist,
#                         personality_factor=2):
#     """Returns a list of recommendation for a given user.
#
#     :param data: a Pandas DataFrame with columns UserId, MovieId, Rating, containing all of the ratings.
#     :param user_id: the id of the active user
#     :param max_recommendations: the maximum number of recommendations that are returned
#     :param num_neighbors: the number of neighbors that the algorithm should consider
#     :param strategy: must be either 'rating' or 'personality'
#     :param p_dist: a DataFrame with columns UserId1, UserId2 and Distance that corresponds to the
#                    personality distances. Must only be provided if strategy is 'personality'
#     :param personality_factor: personality factor for personality-based recommendations.
#     :return: a tuple of the form (movies, ratings) with movies being a list recommended movie ids and
#              ratings being a list of predicted ratings for those movies
#     """
#     assert strategy in ['rating', 'personality'], 'Wrong strategy specifier!'
#     if strategy == 'personality':
#         assert p_dist is not None, 'You must provide a DataFrame with personality scoress!'
#
#     movies = set(data.MovieId.unique())
#     seen = set(data[data.UserId == user_id].MovieId.unique())
#
#     # Discard all movies that the active user has already seen
#     movies.discard(seen)
#
#     if strategy == 'rating-based':
#         y_pred = predict_rating_based(data, movies, user_id, num_neighbors)
#     else:
#         y_pred = predict_personality_based(
#             data=data,
#             p_dist=p_dist,
#             movies=movies,
#             user_id=user_id,
#             num_neighbors=num_neighbors,
#             personality_factor=personality_factor,
#         )
#
#     # Select the movies with the largest predictions
#     df = pd.DataFrame(zip(movies, y_pred), columns=['MovieId', 'PredictedRating']).dropna()
#     recs = df.nlargest(max_recommendations, 'PredictedRating')
#
#     # If we selected too much because of keep=all, drop the less important movies
#     recs = recs.iloc[:max_recommendations]
#
#     movies = list(recs.MovieId)
#     y_pred = list(recs.PredictedRating)
#
#     return movies, y_pred


def partition_by_relevance(movie_ids, y_pred, y_true, relevancy_threshold):
    """Returns a a partition of the relevant and non-relevant predicted movies
       and in addition a list of truly relevant movies.

    :param movie_ids: a list of movie ids
    :param y_pred: the predicted ratings for these movies
    :param y_true: the true ratings for these movies
    :param relevancy_threshold: if a rating is higher than this threshold, it is considered as being relevant
    :return: a tuple of the form (pred_rel, pred_non_rel, true_rel) where the first two elements are a partition
             of the movie ids into a set of (predicted) relevant and non-relevant items and the third element is
             a set of truly relevant items.
    """
    m_pred = list(zip(movie_ids, y_pred))
    m_true = list(zip(movie_ids, y_true))

    pred_rel = [m[0] for m in m_pred if m[1] >= relevancy_threshold]
    pred_non_rel = [m for m in movie_ids if m not in pred_rel]
    true_rel = [m[0] for m in m_true if m[1] >= relevancy_threshold]

    return pred_rel, pred_non_rel, true_rel


def compute_f1_score(pred_rel, pred_non_rel, true_rel):
    """Returns the F1-Score of the given predictions and ground truth labels.

    :param pred_rel: a list of items that are predicted as being relevant
    :param pred_non_rel: a list of items that are predicted as being non-relevant
    :param true_rel: a list of items that are truly relevant
    :return: the F1-Score for the given predictions and ground truths
    """
    # Number of true positives is the size of the intersection
    # between predicted relevant items and true relevant items
    tp = len(set(pred_rel).intersection(set(true_rel)))
    fp = len(pred_rel) - tp

    # Number of false negatives is the size of the intersection
    # between predicted non-relevant items and actually (true) relevant items
    fn = len(set(pred_non_rel).intersection(set(true_rel)))

    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    except ZeroDivisionError:
        return np.nan


def cross_validate(data, demographics, user_id, num_vals, num_neighbors, strategy, demographic_factor, sim_metric,
                   personalities=None, personality_factor=None, relevancy_threshold=6, verbose=False):
    """Returns the cross-validated rmse and f1-score for the given user.

    This function performs cross-validation on for a given user and a given metric.
    Sometimes the algorithm can't predict a rating for a given item because there
    are no ratings in the neighborhood. If this happens for all novel movies, the
    returned metric can be NaN, so keep this in mind.

    :param data: the rating DataFrame with columns UserId, MovieId, Rating
    :param demographics: the demographics DataFrame with columns UserId, CountryId, GenderId and Age
    :param user_id: the id of the active user
    :param num_vals: the number of validations and validation sets for cross-validation
    :param num_neighbors: the maximum number of neighbors that are considered in the algorithms
    :param strategy: either 'rating' or 'personality'
    :param demographic_factor: demographic factor the demographic similarities
    :param sim_metric: The similarity measure that is used to compute the neighborhoods and to serve
                       as a weight in the CF algorithm. Can be either cosine, jaccard, euclidean or pearson,
                       which correspond to the Cosine similarity, Jaccard similarity, inverted Euclidean
                       distance of the Pearson correlation coefficient.
    :param personalities: the personality DataFrame with columns UserId, Extraversion, Agreeableness, Conscientiousness,
                          Neuroticism and Openness
    :param personality_factor: personality factor for personality-based recommendations (default is 2).
    :param relevancy_threshold: the rating threshold for considering a movie as relevant. Movies with
                                ratings equal or higher to this threshold will be considered relevant in
                                the computation of the F1-Score (default is 6).
    :param verbose: prints the individual cross-validation steps if True (default is False)
    :return:the cross-validated rmse and f1-score for the specified user (or maybe NaN)
    """
    assert strategy in ['rating', 'personality'], 'Wrong strategy specifier!'

    if strategy == 'personality':
        assert personalities is not None, 'You must provide a DataFrame with personality scores!'

    splits = np.array_split(data[data.UserId == user_id], num_vals)

    for i in range(num_vals):
        ratings = data.copy()

        # Hold-out set of movies and ground truth ratings
        holdout = list(splits[i].MovieId)
        y_true = splits[i].Rating

        # Remove part of the ratings of the user
        user_ratings = ratings[(ratings.UserId == user_id)]
        user_ratings = user_ratings.query('MovieId not in @holdout')
        ratings[ratings.UserId == user_id] = user_ratings

        if strategy == 'rating':
            y_pred = predict_rating_based(
                data=ratings,
                demographics=demographics,
                movies=holdout,
                user_id=user_id,
                num_neighbors=num_neighbors,
                sim_metric=sim_metric,
                demographic_factor=demographic_factor
            )

            if verbose:
                print(f'---- Rating-based RMSE for user {user_id} (CV {i + 1}/{num_vals}) ----\n')
            rmse = compute_rmse(y_pred, y_true, print_summary=verbose)
            f1_score = compute_f1_score(*partition_by_relevance(holdout, y_pred, y_true, relevancy_threshold))
        else:
            y_pred = predict_personality_based(
                data=ratings,
                personalities=personalities,
                demographics=demographics,
                movies=holdout,
                user_id=user_id,
                num_neighbors=num_neighbors,
                personality_factor=personality_factor,
                demographic_factor=demographic_factor,
                sim_metric=sim_metric
            )

            if verbose:
                print(f'---- Personality-based RMSE for user {user_id} (CV {i + 1}/{num_vals}) ----\n')
            rmse = compute_rmse(y_pred, y_true, print_summary=verbose)
            f1_score = compute_f1_score(*partition_by_relevance(holdout, y_pred, y_true, relevancy_threshold))

    return rmse, f1_score
