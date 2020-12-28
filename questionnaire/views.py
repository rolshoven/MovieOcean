from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from accounts.models import CustomUser
import threading

from .models import IPIPQuestion, IPIPChoice
from .tasks import recompute_neighborhoods

import json


class QuestionnaireView(LoginRequiredMixin, generic.ListView):
    """ Paginated questionnaire view (saves each paginated result to db when proceeding). """
    paginate_by = 5
    template_name = 'questionnaire/ipip50.html'
    context_object_name = 'questions_answers'

    def get_queryset(self):
        """ Fetch all questions and the user's answers if they already exist. """
        questions = IPIPQuestion.objects.all()
        user = self.request.user
        res = []
        for q in questions:
            try:
                # Try to fetch an existing answer
                answer = IPIPChoice.objects.get(question=q, answered_by=user)
                choice = answer.choice
            except IPIPChoice.DoesNotExist:
                # There is no answer to this question yet
                choice = None
            res.append((q, choice))
        return res

    def get_context_data(self, **kwargs):
        """ Add IPIPChoices to context. """
        context = super(QuestionnaireView, self).get_context_data(**kwargs)
        context['choices'] = IPIPChoice.LIKERT_SCALE_CHOICES
        return context

    def post(self, request, *args, **kwargs):
        """ Save choices when clicking through paginated questionnaire. """
        for key in request.POST:
            if key.startswith('question-'):
                question_id = int(key.lstrip('question-'))
                choice = int(request.POST[key])
                question = IPIPQuestion.objects.get(pk=question_id)
                user = request.user
                # Check if user has already answered that question
                try:
                    # If the answer already exsits, change it
                    answer = IPIPChoice.objects.get(question=question, answered_by=user)
                    answer.choice = choice
                except IPIPChoice.DoesNotExist:
                    # This is a new choice, save it to DB
                    answer = IPIPChoice(question=question, answered_by=user, choice=choice)
                answer.save()
        if request.POST['page'] == 'save':
            # Trigger recomputation of neighborhoods
            recompute_neighborhoods.delay()
            # Return the results of the questionnaire
            return HttpResponseRedirect(reverse('questionnaire:results'))
        else:
            # Return to questionnaire with correct pagination
            return HttpResponseRedirect(reverse('questionnaire:index', kwargs={'page': int(request.POST['page'])}))


@login_required
def results(request):
    user = request.user
    answers = IPIPChoice.objects.filter(answered_by=user)

    # If not all questions were answered, redirect to questionnaire with error message
    if len(answers) < 50:
        messages.error(request, f'You only answered {len(answers)} out of 50 questions. Please answer all of them!')
        return HttpResponseRedirect(reverse('questionnaire:index', kwargs={'page': 0}))

    if len(user.get_ratings()) < 10:
        messages.error(request, """In order for this Recommender System to work, we need your ratings. 
                                     Please consider rating at least ten movies to help us improve the 
                                     recommendations. You can use the search function in the menu option 
                                     Movies above to do so.""")

    # User answered all questions, set personality profile flag to true
    user.has_personality_profile = True
    user.save()

    # Compute the personality profile on the fly
    labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    labels_mobile = ['O', 'C', 'E', 'A', 'N']
    scores = user.get_big_five_scores()
    print(scores)
    ordered_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    # Add labels and scores to template context
    context = {
        'labels': json.dumps(labels),
        'mobile': json.dumps(labels_mobile),
        'scores': json.dumps([scores[lb] for lb in labels]),
        'personality': ordered_scores
    }
    return render(request, 'questionnaire/results.html', context)
