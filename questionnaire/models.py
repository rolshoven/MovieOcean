from django.db import models
from django.conf import settings

user_model = settings.AUTH_USER_MODEL


class IPIPQuestion(models.Model):
    SCORED_TRAIT_CHOICES = [
        (1, 'Extraversion'),
        (2, 'Agreeableness'),
        (3, 'Conscientiousness'),
        (4, 'Neuroticism'),
        (5, 'Openness')
    ]
    # Question should appear only once in database
    question_text = models.CharField(max_length=200, unique=True)
    scored_trait = models.IntegerField(choices=SCORED_TRAIT_CHOICES)
    positive_keyed = models.BooleanField()

    def __str__(self):
        return self.question_text


class IPIPChoice(models.Model):
    LIKERT_SCALE_CHOICES = [
        (1, 'Very Inaccurate'),
        (2, 'Moderately Inaccurate'),
        (3, 'Neither Accurate Nor Inaccurate'),
        (4, 'Moderately Accurate'),
        (5, 'Very Accurate')
    ]
    question = models.ForeignKey(IPIPQuestion, on_delete=models.CASCADE)
    answered_by = models.ForeignKey(user_model, on_delete=models.CASCADE)
    choice = models.IntegerField(choices=LIKERT_SCALE_CHOICES)

    class Meta:
        # Only allow one answer per user
        unique_together = ('question', 'answered_by')

    def __str__(self):
        return self.choice
