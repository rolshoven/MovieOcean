from django.contrib import admin
from django.urls import include, path
from accounts.views import home, cookie_info


urlpatterns = [
    path('', home, name='home'),
    path('accounts/', include('accounts.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
    path('questionnaire/', include('questionnaire.urls')),
    path('movies/', include('movie.urls')),
    path('admin/', admin.site.urls),
    path('cookies/', cookie_info, name='cookies')
]
