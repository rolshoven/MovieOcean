from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
import logging
import tmdbsimple as tmdb
from django.utils import timezone

logger = logging.getLogger(__name__)

user_model = settings.AUTH_USER_MODEL

tmdb.API_KEY = settings.TMDB_API_KEY


class Movie(models.Model):
    tmdb_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=256)

    def info(self):
        return tmdb.Movies(self.tmdb_id).info()

    def __str__(self):
        return self.title


class Rating(models.Model):
    class Meta:
        # Only allow one rating per user and movie (can be updated)
        unique_together = ('movie', 'rated_by')

    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rated_by = models.ForeignKey(user_model, on_delete=models.CASCADE, related_name='ratings')
    rating = models.IntegerField(validators=[
        MinValueValidator(1),
        MaxValueValidator(10),
    ])

    def as_stars(self):
        """ Returns the rating as a star rating from 1 to 5 stars (with 0.5 steps) """
        return self.rating / 2

    def __str__(self):
        return f'Rating of {self.rating} for \'{self.movie}\' by {self.rated_by}'


class Recommendation(models.Model):
    class Meta:
        unique_together = ('movie', 'user')

    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user = models.ForeignKey(user_model, on_delete=models.CASCADE, related_name='recommendations')
    created_at = models.DateTimeField(auto_now_add=True)
    is_genre_based = models.BooleanField(default=False)
    genres = models.CharField(max_length=512, default='')
    predicted_star_rating = models.FloatField(
        default=0.0,
        validators=[
            MinValueValidator(0.0),
            MaxValueValidator(5.0)
        ]
    )

    @staticmethod
    def get_recommendations_for_user(user, page=None):
        recommendations = Recommendation.objects.filter(user=user)
        if page:
            # If page is too high, return an empty list
            if 20 * (page -1) > len(recommendations):
                return []
            # Return 20 movies to show on that page
            recommendations = recommendations[20*(page-1):20*(page)]

        if len(recommendations) > 0:
            tmp = recommendations[0]
            recommendations = [{
                'movie': rec.movie.info(),
                'predicted_star_rating': rec.predicted_star_rating
            } for rec in recommendations]
            if tmp.is_genre_based:
                return recommendations, tmp.genres.split(', ')
            return recommendations
        return []

    @staticmethod
    def clear_recommendations_for_user(user):
        recs = user.recommendations.all()
        try:
            for rec in recs:
                rec.delete()
            return True
        except Exception as e:
            logger.exception(f'Clearing recommendations failed with exception: {e}')
            return False

    @staticmethod
    def update_old(user, delay, max_recommendations):
        """ Updates the recommendations for a user if they are older than delay minutes. """
        rec = user.recommendations.first()
        t = timezone.timedelta(minutes=delay)
        if not rec or timezone.now() - rec.created_at > t:
            user.update_cached_recommendations(max_recommendations)
            logger.info(f'Computed recommendations for user {user}.')

    @staticmethod
    def save_recommendations_for_user(user, max_recommendations, update_db=True):
        """
        Computes the up to max_recommendations movie recommendations for the given user.
        If update_db = True, the recommendations will be updated in the DB (make sure to
        check for existing recommendations with the same user-movie tuple first). If
        update_db = False, the recommendations will be created but not saved. You can save
        them individually by looping through the list of Recommendations that this function returns.

        :param user: The user for which the recommendations are computed
        :param max_recommendations: Number of max recommendations that are returned
        :param update_db: Whether or not to save the recommendations to the database
        :return: a list of recommendations for the given user
        """
        recommendations = user.get_recommendations(max_recommendations)
        result = []
        # If we have genre based recommendations
        if len(recommendations) == 2:
            recommendations, genres = recommendations
            genre_string = ', '.join(genres)
            for rec in recommendations:
                rec = Recommendation(
                        user=user,
                        movie=Movie.objects.filter(tmdb_id=rec['id']).first(),
                        is_genre_based=True,
                        genres=genre_string
                )
                result.append(rec)
                if update_db:
                    rec.save()
        else:
            for rec in recommendations:
                rec = Recommendation(
                        user=user,
                        movie=Movie.objects.filter(tmdb_id=rec['movie'].tmdb_id).first(),
                        predicted_star_rating=rec['predicted_star_rating']
                )
                result.append(rec)
                if update_db:
                    rec.save()
        return result

    def __str__(self):
        return f'Movie recommendation for user {self.user}: {self.movie}'


class WatchlistEntry(models.Model):
    class Meta:
        unique_together = ('movie', 'user')
        verbose_name = 'Watchlist Entry'
        verbose_name_plural = 'Watchlist Entries'

    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user = models.ForeignKey(user_model, on_delete=models.CASCADE, related_name='watchlist')

    def __str__(self):
        return f'Watchlist entry for movie \'{self.movie}\' by user \'{self.user}\''
