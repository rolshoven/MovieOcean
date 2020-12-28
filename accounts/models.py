from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser
from django.db import models
from questionnaire.models import IPIPChoice, IPIPQuestion
from django.core.validators import MinValueValidator
import json
import numpy as np
import pandas as pd
import os
from django.conf import settings
from movie.models import Movie, Rating, WatchlistEntry, Recommendation
import tmdbsimple as tmdb
import datetime
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import PermissionsMixin

tmdb.API_KEY = settings.TMDB_API_KEY

DATA_PATH = os.path.join(settings.BASE_DIR, 'data')

path = os.path.join(DATA_PATH, 'countries.json')
with open(path, 'r', encoding='utf-8') as country_list:
    countries = json.load(country_list)

path_neigh = os.path.join(DATA_PATH, 'neighborhood.csv')
path_dist = os.path.join(DATA_PATH, 'distances.csv')


def is_adult(date_of_birth):
    if (datetime.date.today() - date_of_birth) < datetime.timedelta(days=18 * 365):
        raise ValidationError(
            _("""
                Because we do not filter adult movies, you have to be 18 years 
                old or older to use this website. We\'re sorry!
            """),
            params={'date_of_birth': date_of_birth},
        )


def get_genres_for_user(user, max_distance=2.5, max_genres=4):
    # Get mean genre personality vector from the results of Cantador et al.
    genre_centroids = pd.read_csv(os.path.join(DATA_PATH, 'genre_personalities.csv'))

    # Get personality vector of user and rescale it to match the centroids above
    scores = user.get_big_five_scores()
    user_personality = np.array([scores[dim] / 10 for dim in genre_centroids.columns[1:]])

    # Compute euclidean distances between user personality and genre centroid in personality space
    genre_centroids['distance'] = np.linalg.norm(genre_centroids.iloc[:, 1:] - user_personality, axis=1)

    # Exclude genres that are too distant
    genre_centroids = genre_centroids[genre_centroids['distance'] <= max_distance]

    # Sort genres by ascending distance to user personality
    genre_centroids = genre_centroids.sort_values('distance')

    return list(genre_centroids['Genre'])[:max_genres]


def nonzero_ratings_mean(rating_matrix):
    """
    Returns the mean rating vector for given set of users ignoring ratings that are zero (not rated items)

    :param rating_matrix: A matrix with shape (m, n), where m is the number of users and n is the number of
                          different items that were rated at least once in the neighborhood of these users.
    :return: A vector of shape (m,) containing the mean rating for each user
    """
    tmp = rating_matrix.copy()
    # Set zeros zo NaN
    tmp[tmp == 0] = np.nan
    # If it is just a rating vector with shape (n,), it should also work
    return np.nanmean(tmp, axis=int(len(tmp.shape) == 2))


def rating_deviations(rating_matrix):
    """
    Computes the deviations of the user ratings from their mean rating.

    :param rating_matrix: A matrix with shape (m, n), where m is the number of users and n is the number of
                          different items that were rated at least once in the neighborhood of these users.

    :return:              A matrix with the same shape as rating_matrix but with the mean rating subtracted for each
                          rated item.
    """
    tmp = rating_matrix.copy()
    not_rated_indices = np.where(tmp == 0)
    tmp = (tmp.T - nonzero_ratings_mean(tmp)).T
    tmp[not_rated_indices] = 0
    return tmp


def skip_neighbors(active_user, min_novelty, neighbors, distances, neighborhoods):
    """
    Removes users from neighborhood if they do not provide enough new movie ratings.
    :param active_user: the user for which the neighborhood was computed.
    :param neighbors: the neighbors that were computed.
    :param min_novelty: the minimum number of new movie ratings that a user should have compared to the active user.
    :param distances: the precomputed distances to the other users.
    :param neighborhoods: the precomputed neighborhood as a user-suer matrix.
    :return: The new list of neighbors.
    """
    skip_user_ids = []
    active_user_rated_movies = active_user.get_rated_movies()

    for user in neighbors:
        user_movies = [m for m in user.get_rated_movies() if m not in active_user_rated_movies]
        if len(user_movies) < min_novelty:
            skip_user_ids.append(user.id)
    neighbors = [u for u in neighbors if u.id not in skip_user_ids]
    neighbor_ids = [u.id for u in neighbors]

    # If there are users that were skipped, set the next likely neighborhood (by distance)
    num_missing = settings.NEIGHBORHOOD_SIZE - len(neighbors)
    dists = distances[str(active_user.id)].drop([active_user.id, *neighbor_ids, *skip_user_ids]).sort_values()
    while len(dists) > num_missing > 0:
        # Compute new user ids
        new_user_ids = list(dists.nsmallest(len(skip_user_ids)).index)
        neighborhoods[str(active_user.id)][skip_user_ids] = np.zeros_like(skip_user_ids)  # set skipped users to zero
        neighborhoods[str(active_user.id)][new_user_ids] = 1  # Add new users to neighborhood

        # Get neighbors from database
        neighbor_ids = list(neighborhoods[str(active_user.id)][neighborhoods[str(active_user.id)] == 1].index)
        neighbors = list(CustomUser.objects.filter(pk__in=neighbor_ids))

        # Check if novelty criterion is fulfilled, remove users that did not rate enough new movies
        for user in neighbors:
            user_movies = [m for m in user.get_rated_movies() if m not in active_user_rated_movies]
            if len(user_movies) < min_novelty:
                skip_user_ids.append(user.id)
        neighbors = [u for u in neighbors if u.id not in skip_user_ids]
        neighbor_ids = [u.id for u in neighbors]

        # Update number of missing users in neighborhood
        num_missing = settings.NEIGHBORHOOD_SIZE - len(neighbors)
        dists = distances[str(active_user.id)].sort_values().drop([active_user.id, *neighbor_ids, *skip_user_ids])

    return neighbors


class CustomUserManager(BaseUserManager):
    """
    Custom user model manager where email is the unique identifiers
    for authentication instead of usernames.

    Credits: https://tech.serhatteker.com/post/2020-01/email-as-username-django/
    """

    def create_user(self, email, date_of_birth, gender, country, password, **extra_fields):
        """
        Create and save a User with the given email and password.
        """
        if not email:
            raise ValueError(_('The Email must be set'))
        if not date_of_birth:
            raise ValueError(_('The date of birth be set'))
        if not gender:
            raise ValueError(_('The gender must be set'))
        if not country:
            raise ValueError(_('The country must be set'))

        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.date_of_birth = date_of_birth
        user.gender = gender
        user.country = country
        user.save()
        return user

    def create_superuser(self, email, date_of_birth, gender, country, password, **extra_fields):
        """
        Create and save a SuperUser with the given email and password.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('is_admin', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))

        return self.create_user(email, date_of_birth, gender, country, password, **extra_fields)


class CustomUser(AbstractBaseUser, PermissionsMixin):
    GENDER_CHOICES = [
        (1, 'Female'),
        (2, 'Male'),
        (3, 'Transgender Female'),
        (4, 'Transgender Male'),
        (5, 'Gender Variant/Non-Conforming'),
        (6, 'Not Listed'),
        (7, 'Prefer Not to Answer')
    ]
    COUNTRY_CHOICES = [(i + 1, country) for i, country in enumerate([c['name'] for c in countries])]

    username = None
    email = models.EmailField(_('email address'), unique=True)
    date_of_birth = models.DateField(_('date of birth'), default=None, validators=[is_adult])
    gender = models.IntegerField(_('gender'), choices=GENDER_CHOICES)
    country = models.IntegerField(_('country'), choices=COUNTRY_CHOICES)
    date_joined = models.DateTimeField(auto_now_add=True)

    has_personality_profile = models.BooleanField(_('Has personality profile'), default=False)
    rating_reminders = models.BooleanField(_('Receives rating reminders'), default=True)

    is_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    first_name = models.CharField(max_length=128, default='')
    last_name = models.CharField(max_length=128, default='')

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = [date_of_birth, gender, country]

    objects = CustomUserManager()

    def get_big_five_scores(self):
        answers = IPIPChoice.objects.filter(answered_by=self)
        if len(answers) < 50:
            raise Exception(f'User {self} did only answer {len(answers)} out of 50 questions!')
        labels = [c[1] for c in IPIPQuestion.SCORED_TRAIT_CHOICES]
        scores = {trait: 0 for trait in labels}
        for answer in answers:
            q = answer.question
            trait = q.get_scored_trait_display()
            scores[trait] += answer.choice if q.positive_keyed else 6 - answer.choice
        return scores

    def get_ratings(self):
        return Rating.objects.filter(rated_by=self)

    def get_rated_movies(self):
        return [r.movie for r in self.get_ratings()]

    # calculates neighborhood matrix and saves it in ./data/neighborhood.csv
    # call with CustomUser.calculate_neighborhoods()
    @staticmethod
    def calculate_neighborhoods(k=settings.NEIGHBORHOOD_SIZE):
        # go through scores of other users and add max k users to neighborhood
        all_users = CustomUser.objects.filter(has_personality_profile=True)

        user_ids = list(all_users.values_list('id', flat=True))

        neighborhoods = pd.DataFrame(np.zeros((len(all_users), len(all_users))), user_ids, user_ids)
        distances = pd.DataFrame(np.zeros((len(all_users), len(all_users))), user_ids, user_ids)

        for current_user in all_users:
            # get scores of current user
            current_user_scores = current_user.get_big_five_scores()
            current_user_personality = np.array(list(current_user_scores.values()))

            # neighborhood of current user
            for user in all_users:
                if user != current_user:
                    # calculate euclidean distance of personalities
                    user_scores = user.get_big_five_scores()
                    user_personality = np.array(list(user_scores.values()))
                    distance = np.linalg.norm(current_user_personality - user_personality)
                    # add to list
                    distances[current_user.id][user.id] = distance  # columns = current_users

            # Compute neighborhood
            most_similar_rows = list(distances[current_user.id].nsmallest(k + 1).index)
            neighborhoods[current_user.id][most_similar_rows] = 1
            neighborhoods[current_user.id][current_user.id] = 0

        neighborhoods.to_csv(path_neigh)
        distances.to_csv(path_dist)

        return neighborhoods

    @staticmethod
    def load_neighborhoods(k=settings.NEIGHBORHOOD_SIZE):
        if not os.path.exists(path_neigh):
            CustomUser.calculate_neighborhoods(k)
        return pd.read_csv(path_neigh, index_col=0)

    @staticmethod
    def load_distances(k=settings.NEIGHBORHOOD_SIZE):
        if not os.path.exists(path_neigh):
            CustomUser.calculate_neighborhoods(k)
        return pd.read_csv(path_dist, index_col=0)

    def get_neighborhood_and_distances(self, min_novelty=3, min_users=2, k=settings.NEIGHBORHOOD_SIZE):
        """
        Returns the neighborhood and the distances from the active user to each of the users in the neighborhood.
        This function tries to only add users to the neighborhood who add at least min_novelty movie ratings
        compared to the active user. However, it also ensures that at least min_users are present in the neighborhood.
        This is important in the case where no user adds enough new movies to the neighborhood. Otherwise, the returned
        neighborhood would always be empty.

        :param k: size of the neighborhood
        :param min_novelty: the minimum number of new movies that a neighbor has compared to the active user.
        :param min_users: the minimum number of users that the neighborhood should have.
        :return: a tuple of the form (neighborhood, distances)
        """
        # Load precomputed neighborhoods and distances
        neighborhoods = self.load_neighborhoods(k)
        distances = self.load_distances(k)

        # Get neighbors from database
        try:
            neighbor_ids = list(neighborhoods[str(self.id)][neighborhoods[str(self.id)] == 1].index)
        except KeyError:
            # User not in precomputed neighborhood, recompute!
            neighborhoods = CustomUser.calculate_neighborhoods()
            distances = CustomUser.load_distances()
            neighbor_ids = list(neighborhoods[str(self.id)][neighborhoods[str(self.id)] == 1].index)
        neighbors = list(CustomUser.objects.filter(pk__in=neighbor_ids))

        neighbors_new = skip_neighbors(self, min_novelty, neighbors.copy(), distances.copy(), neighborhoods.copy())

        # If we filtered out too many users, add some users again until we have min_users in the neighborhood.
        n_idx = 0
        while len(neighbors_new) < min_users and n_idx < len(neighbors):
            if neighbors[n_idx] not in neighbors_new:
                neighbors_new.append(neighbors[n_idx])
            n_idx += 1

        neighbors_new_ids = [u.id for u in neighbors_new]

        return neighbors_new, distances[str(self.id)][neighbors_new_ids]

    def get_recommendations(self, max_recommendations, personality_factor=2, user_cf_threshold=10,
                            mean_distance_threshold=35, eps=1e-5, k=settings.NEIGHBORHOOD_SIZE):
        # Get neighborhood and distances
        neighborhood, distances = self.get_neighborhood_and_distances(k)

        # If we have less than user_cf_threshold users and the mean distance in the neighborhood is high,
        # use personality-genre based recommendations
        if CustomUser.objects.all().count() <= user_cf_threshold:
            if distances.mean() > mean_distance_threshold:
                return self.get_genre_based_recommendations(max_recommendations)

        # Get Ratings of neighbors and active user
        ratings = {user.id: list(Rating.objects.filter(rated_by=user)) for user in neighborhood}
        r_au = list(Rating.objects.filter(rated_by=self))

        # The order of the keys will define the order of the users in the rating matrix
        user_ids = ratings.keys()

        # Get all movies that were rated (but only once)
        movies = set()
        for rating_list in ratings.values():
            for rating in rating_list:
                movies.add(rating.movie)

        # If there are no new movies in the neighborhood, use genre-based recommendations
        if len(movies) == 0:
            return self.get_genre_based_recommendations(max_recommendations)

        # Add movies of active user as well
        for rating in r_au:
            movies.add(rating.movie)

        # Convert to list (so that there exists an order)
        movies = list(movies)

        # rows = users, columns = movies, value = rating
        rating_matrix = np.zeros((len(neighborhood), len(movies)))

        # Fill rating matrix with movie ratings of neighbors
        for row, user_id in enumerate(user_ids):
            for rating in ratings[user_id]:
                col = movies.index(rating.movie)
                rating_matrix[row, col] = rating.rating

        # rating vector of active user
        ratings_au = np.zeros(len(movies))

        # Fill the rating vector of the active user (which should be of the same size as a row in the rating matrix).
        for rating in r_au:
            col = movies.index(rating.movie)
            ratings_au[col] = rating.rating

        # Compute cosine similarity of user ratings and personality similarity (inverted euclidean distance)
        cos_rating_sim = rating_matrix.dot(ratings_au) / (
                np.linalg.norm(ratings_au) * np.linalg.norm(rating_matrix, axis=1) + eps)
        euclidean_pers_sim = 1 / (distances.values + eps)
        euclidean_pers_sim /= (np.linalg.norm(euclidean_pers_sim) + eps)

        # Compute rating predictions using the collaborative filtering algorithm
        sim = (cos_rating_sim + personality_factor * euclidean_pers_sim) / (1 + personality_factor)
        au_mean_rating = nonzero_ratings_mean(ratings_au)
        neighbor_rating_dev = rating_deviations(rating_matrix)
        predicted_ratings = au_mean_rating + np.sum((neighbor_rating_dev.T * sim).T, axis=0) / (np.sum(sim) + eps)

        # Set predicted ratings of movies that the active user has already rated to -1
        rated_movie_indices = [movies.index(r.movie) for r in r_au]
        for i in rated_movie_indices:
            predicted_ratings[i] = -1

        # Get the largest predictions
        n_rec = max_recommendations if max_recommendations <= len(movies) else len(movies)
        movie_indices = np.argpartition(predicted_ratings, -n_rec)[-n_rec:]
        movie_ratings = predicted_ratings[movie_indices]

        # Sort indices and ratings by rating (descending)
        movie_ratings, movie_indices = zip(*sorted(zip(movie_ratings, movie_indices), reverse=True))

        # Extract movies and make sure that no movies are included that the active user already rated
        # Note that this could only happen in special cases, but we want to prevent that anyway.
        recommendations = [movies[i] for i in movie_indices if i not in rated_movie_indices]

        results = [
            {
                'movie': movie,
                'predicted_star_rating': rating / 2,
            } for movie, rating in zip(recommendations, movie_ratings)
        ]

        return results

    def get_genre_based_recommendations(self, max_recommendations):
        """
        Returns movie recommendations solely based on personality-genre associations.
        Returns the recommendations as well as the list of genres for this user.
        """
        genres = get_genres_for_user(self, max_distance=2, max_genres=4)

        genre_ids = [g['id'] for g in tmdb.Genres().movie_list()['genres'] if g['name'] in genres]

        recommendations = []

        args = {
            'page': 1,
            'vote_count_gte': 20,
            'vote_average_gte': 7,
            'sort_by': 'vote_average.desc'
        }

        # If at least one rule can be applied, predict movies of the associated genres with a high rating.
        # Otherwise, just predict movies of all genres with high ratings.
        if len(genres) > 0:
            args['with_genres'] = genre_ids

        recs = tmdb.Discover().movie(**args)
        recommendations += recs['results']
        rated_movie_ids = [r.movie.tmdb_id for r in Rating.objects.filter(rated_by=self)]

        # Filter out movies that the user has already seen
        recommendations = [r for r in recommendations if r['id'] not in rated_movie_ids]

        while len(recommendations) < max_recommendations and recs['page'] < recs['total_pages']:
            args['page'] += 1
            recs = tmdb.Discover().movie(**args)
            recommendations += recs['results']
            # Filter out movies that the user has already seen
            recommendations = [r for r in recommendations if r['id'] not in rated_movie_ids]

        if len(recommendations) > max_recommendations:
            recommendations = recommendations[:max_recommendations]

        # Save recommended Movies in DB
        for rec in recommendations:
            movie = Movie.objects.filter(tmdb_id=rec['id'])
            if not movie.exists():
                Movie(tmdb_id=rec['id'], title=rec['title']).save()

        return recommendations, genres

    def get_cached_recommendations(self, page=None):
        return Recommendation.get_recommendations_for_user(self, page)

    def update_cached_recommendations(self, max_recommendations):
        recs = Recommendation.save_recommendations_for_user(
            user=self,
            max_recommendations=max_recommendations,
            update_db=False
        )
        Recommendation.clear_recommendations_for_user(user=self)
        for rec in recs:
            rec.save()

    def __str__(self):
        return self.email
