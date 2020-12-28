from django.http import HttpResponseRedirect
from django.contrib import messages
from django.urls import reverse
from functools import wraps


def personality_profile_required(function):
    """
    Redirects the user to the questionnaire if there is no personality profile yet.

    For view functions, use the following decorator:
        @personality_profile_required.

    For class-based view, decorate the class as follows:
        @method_decorator(personality_profile_required, name='dispatch')
        (See https://docs.djangoproject.com/en/3.1/topics/class-based-views/intro/#id1)
    """
    @wraps(function)
    def wrap(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseRedirect(reverse('login'))
        elif request.user.has_personality_profile:
            return function(request, *args, **kwargs)
        else:
            messages.error(request, f'Please complete the questionnaire to have access to the Recommender System.')
            return HttpResponseRedirect(reverse('questionnaire:index', kwargs={'page': 0}))
    return wrap
