from django.urls import path
from . import views

app_name = 'questionnaire'
urlpatterns = [
    # ex: /questionnaire/0 (first page of questionnaire)
    path('<int:page>/', views.QuestionnaireView.as_view(), name='index'),
    path('results/', views.results, name='results')
]