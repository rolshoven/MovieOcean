# accounts/urls.py
from django.urls import path

from . import views


urlpatterns = [
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('password_change/', views.password_change, name='password_change'),
    path('password_reset/done/', views.password_reset_done, name='password_reset_done'),
    path('me/', views.AccountsView.as_view(), name='account'),
    path('delete/', views.delete_account, name='delete_account'),
    path('me/reminders', views.toggle_rating_reminders, name='toggle_rating_reminders'),
    path('help/', views.HelpView.as_view(), name='help')
]
