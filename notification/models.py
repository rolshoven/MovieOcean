from django.db import models
from accounts.models import CustomUser
from django.utils import timezone


class Notification(models.Model):
    sent_to = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    type = models.CharField(max_length=256)
    date = models.DateTimeField()

    def save(self, *args, **kwargs):
        """ On save, update timestamp. """
        if not self.id:
            self.date = timezone.now()
        return super(Notification, self).save(*args, **kwargs)

    def __str__(self):
        return f'{self.type} notification for {self.sent_to}'
