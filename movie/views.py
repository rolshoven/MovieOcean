from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils.decorators import method_decorator
from django.conf import settings
from MovieOCEAN.decorators import personality_profile_required
from django.views.generic import TemplateView
from django.http import Http404
from requests.exceptions import HTTPError
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from .models import Movie, Rating, Recommendation, WatchlistEntry
from django.shortcuts import redirect
from .tasks import recompute_recommendations
import logging


from datetime import date
import tmdbsimple as tmdb
import math

logger = logging.getLogger(__name__)

tmdb.API_KEY = settings.TMDB_API_KEY
search = tmdb.Search()


@method_decorator(personality_profile_required, name='dispatch')
class MovieSearchView(LoginRequiredMixin, TemplateView):
    template_name = 'movie/search.html'

    def get(self, request, query=None, page=None, *args, **kwargs):
        query = query if query else request.GET.get('query', None)
        page = page if page else request.GET.get('page', None)
        if query:
            if page:
                response = search.movie(query=query, page=page)
            else:
                response = search.movie(query=query)
            context = {**response, 'query': query}
            for i, movie in enumerate(context['results']):
                movie_reference = Movie.objects.filter(tmdb_id=movie['id']).first()
                if movie_reference:
                    user_rating = Rating.objects.filter(rated_by=request.user, movie=movie_reference).first()
                    if user_rating:
                        context['results'][i]['user_star_rating'] = user_rating.rating / 2
            return render(request, self.template_name, context)
        return render(request, self.template_name)


@method_decorator(personality_profile_required, name='dispatch')
class MovieDetailView(LoginRequiredMixin, TemplateView):
    template_name = 'movie/details.html'

    def get(self, request, id=None, *args, **kwargs):
        if not id:
            raise Http404('No movie id provided.')
        try:
            m = tmdb.Movies(id)
            m.info()  # Triggers exception if movie was not found, otherwise adds info
        except HTTPError:
            raise Http404('Movie could not be found.')
        movie = {
            'id': m.id,
            'title': m.title,
            'genres': [g['name'] for g in m.genres],
            'release_date': date.fromisoformat(m.release_date).strftime('%B %d, %Y'),
            'runtime': m.runtime,
            'production_countries': [c['name'] for c in m.production_countries],
            'poster_path': m.poster_path,
            'overview': m.overview,
            'keywords': [k['name'] for k in m.keywords()['keywords']],
            'cast': [{
                'character': c['character'],
                'name': c['name'],
                'profile_path': c['profile_path'],
            } for c in m.credits()['cast']]
        }
        similar = m.similar_movies()
        if similar:
            similar_movies = [{
                'id': s['id'],
                'title': s['title'],
                'poster_path': s['poster_path'],
            } for s in similar['results'][:12]]  # Only consider up to 12 movies
            movie['similar_movies'] = similar_movies
        if m.original_title and m.original_language and m.original_title != m.title:
            movie['original_title'] = m.original_title
            movie['original_language'] = m.original_language
        if 'results' in m.videos():
            # Get the first YouTube link if there exists one
            yt_keys = [v['key'] for v in m.videos()['results'] if
                       v['iso_639_1'] == 'en' and v['site'] == 'YouTube' and 'trailer' in v['name'].lower()]
            if yt_keys:
                movie['yt_key'] = yt_keys[0]
        if m.homepage:
            movie['homepage'] = m.homepage
        movie_reference = Movie.objects.filter(tmdb_id=movie['id']).first()
        if movie_reference:
            user_rating = Rating.objects.filter(rated_by=request.user, movie=movie_reference).first()
            if user_rating:
                movie['user_star_rating'] = user_rating.rating / 2
        return render(request, self.template_name, {'movie': movie})


@require_http_methods(['POST'])
@login_required
@personality_profile_required
def rate_movie(request):
    """ Rates a movie (this is not actually a view but it can be used in AJAX requests). """
    response = {}

    rating = float(request.POST.get('rating'))
    tmdb_id = int(request.POST.get('tmdb_id'))
    movie_title = request.POST.get('movie_title')

    # Get Movie
    movie, _ = Movie.objects.get_or_create(
        tmdb_id=tmdb_id,
        defaults={'title': movie_title}
    )

    # Convert 1-5 Star rating to 1-5 rating
    rating *= 2

    # Check if rating already exists. If it does, update it.
    rating_queryset = Rating.objects.filter(movie=movie, rated_by=request.user)
    if rating_queryset.exists():
        movie_rating = rating_queryset.first()
        previous_rating = movie_rating.rating
        movie_rating.rating = rating
        message = 'Successfully updated rating.'
    else:
        movie_rating = Rating(movie=movie, rated_by=request.user, rating=rating)
        message = 'Succesfully created new rating.'

    # Validate rating (must be between 0.5 Stars and 5 Stars)
    try:
        movie_rating.clean_fields()
        movie_rating.save()

        # If recommendation exists, delete it
        rec_query = Recommendation.objects.filter(user=request.user, movie=movie)
        if rec_query.exists():
            rec_query.first().delete()

        response['status'] = message
    except Exception as e:
        response['errors'] = 'Ratings must be between 0.5 and 5 stars!'
        response['previous_rating'] = previous_rating / 2 if previous_rating else 0
    finally:
        return JsonResponse(response)


@require_http_methods(['POST'])
@login_required
@personality_profile_required
def add_to_watchlist(request):
    """
    Adds a movie to the watchlist
    (this is not actually a view but it can be used in AJAX requests).
    """
    tmdb_id = int(request.POST.get('tmdb_id'))
    movie = Movie.objects.filter(tmdb_id=tmdb_id)

    if movie.exists():
        movie = movie.first()
    else:
        title = tmdb.Movies(tmdb_id).title
        movie = Movie(tmdb_id=tmdb_id, title=title)
        movie.save()

    watchlist_entry = WatchlistEntry.objects.filter(user=request.user, movie=movie)
    if watchlist_entry.exists():
        return JsonResponse({'errors': f'Watchlist entry for movie {movie} already exists!'})

    WatchlistEntry(user=request.user, movie=movie).save()

    return JsonResponse({'status': f'Successfully added movie \'{movie}\' to watchlist.'})


@require_http_methods(['POST'])
@login_required
@personality_profile_required
def remove_from_watchlist(request):
    """
    Removes a movie from the watchlist.
    """
    tmdb_id = int(request.POST.get('tmdb_id'))
    movie = Movie.objects.filter(tmdb_id=tmdb_id).first()
    wle = WatchlistEntry.objects.filter(user=request.user, movie=movie)

    if wle.exists() > 0:
        wle.first().delete()
        messages.info(request, f'Successfully removed movie \'{movie.title}\' from watchlist.')

    return redirect('movie:watchlist')


@method_decorator(personality_profile_required, name='dispatch')
class RecommendationsView(LoginRequiredMixin, TemplateView):
    template_name = 'movie/recommendations.html'

    def get(self, request, page=None, *args, **kwargs):
        page = page if page else request.GET.get('page', 1)

        try:

            recommendations = request.user.get_cached_recommendations(page=page)

            if page == 1:
                # Trigger recomputation of recommendations if necessary
                recompute_recommendations.delay(
                    user_id=request.user.id,
                    delay=15,
                    max_recommendations=40
                )

            # Movie recommendations were based on genre because the mean distance in neighborhood is too large
            if len(recommendations) == 2:
                recommendations, genres = recommendations

                # User personality did not match any of the 9 genres
                if len(genres) == 0:
                    messages.info(
                        request,
                        """
                            Currently, we do not have a lot of users and we detected that there are not enough users
                            that match your personality. Therefore, the recommendations you see are only based on 
                            the average rating on the website TMDB. If you check your recommendations later and this
                            message does not appear anymore, we will have collected enough users to give you some more
                            meaningful recommendations.
                        """
                    )
                else:
                    messages.info(
                        request,
                        f"""
                            Currently, we do not have a lot of users and we detected that there are not enough users
                            that match your personality. Therefore, the recommendations you see are only based on 
                            the average rating on the website TMDB. Based on your personality, we found that you 
                            could be interested in the following genres: {', '.join(genres)}. Every movie you see 
                            below has at least one of those genres attached. If you check your recommendations later 
                            and this message does not appear anymore, we will have collected enough users to give 
                            you some more meaningful recommendations.
                        """
                    )

            total_results = len(request.user.recommendations.all())

            # 20 results per page
            total_pages = math.ceil(total_results / 20)

            # create response with all necessary attributes for the recommended movies
            response = {'page': page, 'total_results': total_results,
                        'total_pages': total_pages, 'recommendations': recommendations}

            return render(request, self.template_name, response)
        except KeyError:
            return render(request, self.template_name)


@method_decorator(personality_profile_required, name='dispatch')
class WatchlistView(LoginRequiredMixin, TemplateView):
    template_name = 'movie/watchlist.html'

    def get(self, request, page=None, *args, **kwargs):
        page = page if page else request.GET.get('page', 1)

        watchlist = request.user.watchlist.all()
        watchlist = [wle.movie.info() for wle in watchlist[20*(page-1):20*page]]

        total_results = len(watchlist)
        # 20 results per page
        total_pages = math.ceil(total_results / 20)
        # create response with all necessary attributes for the recommended movies
        response = {'page': page, 'total_results': total_results,
                    'total_pages': total_pages, 'movies': watchlist}

        return render(request, self.template_name, response)


class AboutView(LoginRequiredMixin, TemplateView):
    template_name = 'about/about.html'
