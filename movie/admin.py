from django.contrib import admin

from .models import Movie, Recommendation, Rating, WatchlistEntry
from django.contrib.admin import ModelAdmin, BooleanFieldListFilter, DateFieldListFilter
from admin_numeric_filter.admin import NumericFilterModelAdmin, SliderNumericFilter


class RatingAdmin(ModelAdmin):
    model = Movie

    list_display = ('movie', 'rated_by', 'rating')
    list_filter = ('rating',)
    search_fields = ('movie__title', 'rated_by__email',)
    ordering = ('rated_by',)


class WatchlistEntryAdmin(ModelAdmin):
    model = WatchlistEntry

    list_display = ('user', 'movie', )
    search_fields = ('user__email', 'movie__title',)
    ordering = ('user',)


class RecommendationAdmin(NumericFilterModelAdmin):
    model = Recommendation

    list_display = ('user', 'movie', 'created_at', 'predicted_star_rating', 'is_genre_based')
    list_filter = (
        ('created_at', DateFieldListFilter),
        ('is_genre_based', BooleanFieldListFilter),
        ('predicted_star_rating', SliderNumericFilter),
    )
    search_fields = ('movie__title', 'user__email', 'genres')
    ordering = ('-predicted_star_rating',)


# Register your models here.
admin.site.register(Movie)
admin.site.register(Recommendation, RecommendationAdmin)
admin.site.register(Rating, RatingAdmin)
admin.site.register(WatchlistEntry, WatchlistEntryAdmin)
