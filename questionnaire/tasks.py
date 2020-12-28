from accounts.models import CustomUser

from celery import shared_task


@shared_task
def recompute_neighborhoods():
    CustomUser.calculate_neighborhoods()

