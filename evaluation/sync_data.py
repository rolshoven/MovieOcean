"""
This file synchronizes the data from the database and stores it in csv files for further
evaluation. The user ids and movie ids will be randomized so that there is no way to
establish a connection between certain users and certain movies/ratings.
"""

# Set up django to work in this stand-alone script
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MovieOCEAN.settings')
django.setup()

# Other imports
import random
import pandas as pd
import numpy as np
from accounts.models import CustomUser
from movie.models import Movie, Rating
from utils import get_rating_distances
from django.conf import settings
import tmdbsimple as tmdb
from datetime import date
from requests.exceptions import HTTPError

tmdb.API_KEY = settings.TMDB_API_KEY

# Get all movie and user ids from the database
movie_ids = list(Movie.objects.all().values_list('id', flat=True))
user_ids = list(CustomUser.objects.filter(has_personality_profile=True).values_list('id', flat=True))
country_ids = [cc[0] for cc in CustomUser.COUNTRY_CHOICES]
gender_ids = [gc[0] for gc in CustomUser.GENDER_CHOICES]

# Translate original ids into random ids
random.shuffle(movie_ids)
random.shuffle(user_ids)
random.shuffle(country_ids)
random.shuffle(gender_ids)
translate_mid = {id: i for i, id in enumerate(movie_ids)}
translate_uid = {id: i for i, id in enumerate(user_ids)}
translate_cid = {id: i for i, id in enumerate(country_ids)}
translate_gid = {id: i for i, id in enumerate(gender_ids)}

# Store ratings in a csv file
rating_data = []
ratings = Rating.objects.all()
for r in ratings:
    try:
        genres = ';'.join([g['name'] for g in tmdb.Movies(r.movie.tmdb_id).info()['genres']])
    except HTTPError:
        genres = 'Unknown'
    rating_data.append([
        translate_uid[r.rated_by.id],
        translate_mid[r.movie.id],
        r.rating,
        genres
    ])
df_ratings = pd.DataFrame(rating_data, columns=['UserId', 'MovieId', 'Rating', 'Genres'])
df_ratings.to_csv(os.path.join('data', 'ratings.csv'), index=False)

# Store personalities and demographic data in csv files
personality_data = []
demographic_data = []
traits = [
    'Extraversion',
    'Agreeableness',
    'Conscientiousness',
    'Neuroticism',
    'Openness',
]
today = date.today()
for uid in user_ids:
    user = CustomUser.objects.get(pk=uid)
    scores = user.get_big_five_scores()
    scores = [scores[trait] for trait in traits]
    personality_data.append([translate_uid[uid], *scores])
    age = (today - user.date_of_birth).days // 365
    demographic_data.append([translate_uid[uid], translate_cid[user.country], translate_gid[user.gender], age])

df_personality = pd.DataFrame(personality_data, columns=['UserId', *traits])
df_demographics = pd.DataFrame(demographic_data, columns=['UserId', 'CountryId', 'GenderId', 'Age'])
df_personality.to_csv(os.path.join('data', 'personalities.csv'), index=False)
df_demographics.to_csv(os.path.join('data', 'demographics.csv'), index=False)

# Compute rating distances between users and store them in a csv file
df_rating_distances = get_rating_distances(df_ratings)
df_rating_distances.to_csv(os.path.join('data', 'rating_distances.csv'), index=False)

# Compute personality distances and store them in a csv file
personality_distance_data = []
for i in range(len(df_personality) - 1):
    for j in range(i+1, len(df_personality)):
        u1 = df_personality.iloc[i].UserId
        u2 = df_personality.iloc[j].UserId
        u1_personality = df_personality.iloc[i, 1:].values
        u2_personality = df_personality.iloc[j, 1:].values
        distance = np.linalg.norm(u1_personality - u2_personality)
        personality_distance_data.append([u1, u2, distance])
df_personality_distances = pd.DataFrame(personality_distance_data, columns=['UserId1', 'UserId2', 'Distance'])
df_personality_distances.to_csv(os.path.join('data', 'personality_distances.csv'), index=False)
