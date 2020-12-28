from django.urls import path
from . import views

app_name = 'movie'
urlpatterns = [
    path('search/', views.MovieSearchView.as_view(), name='search'),
    path('search/<str:query>/<int:page>/', views.MovieSearchView.as_view(), name='search'),
    path('view/<int:id>/', views.MovieDetailView.as_view(), name='view'),
    path('rate/', views.rate_movie, name='rate'),
    path('recommendations/', views.RecommendationsView.as_view(), name='recommendations'),
    path('recommendations/<int:page>/', views.RecommendationsView.as_view(), name='recommendations'),
    path('watchlist/', views.WatchlistView.as_view(), name='watchlist'),
    path('watchlist/<int:page>/', views.WatchlistView.as_view(), name='watchlist'),
    path('watchlist/add/', views.add_to_watchlist, name='add_to_watchlist'),
    path('watchlist/remove/', views.remove_from_watchlist, name='remove_from_watchlist'),
    path('about/', views.AboutView.as_view(), name='about'),
]