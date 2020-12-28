from django.core.management.base import BaseCommand, CommandError
from accounts.models import CustomUser
from notification.models import Notification
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
import os

msg_path = os.path.join(settings.BASE_DIR, 'data', 'rating_reminder_message.txt')


class Command(BaseCommand):
    help = 'After {days=7} days after registration, send out reminders to user to rate the recommendations.'

    def add_arguments(self, parser):
        parser.add_argument('days', nargs='?', type=int, default=7)

    def handle(self, *args, **options):
        try:
            with open(msg_path, 'r', encoding='utf-8') as file:

                message = file.read()
                users = CustomUser.objects.all()
                notified_users = [n.sent_to for n in Notification.objects.filter(type='RATING_REMINDER')]

                # Only send to users that want to receive mails
                users = [u for u in users if u.rating_reminders]

                # Only send to users that have not yet received the mail
                users = [u for u in users if u not in notified_users]

                # Users must be registered for at least 7 days
                now = timezone.now()
                days = options['days']
                users = [u for u in users if (now - u.date_joined).days >= days]

                for user in users:
                    if user.has_personality_profile:
                        send_mail(
                            'Have you seen our recommendations?',
                            message,
                            'noreply@movieocean.de',
                            [user.email],
                            fail_silently=False
                        )

                        Notification(sent_to=user, type='RATING_REMINDER').save()
        except Exception as e:
            self.stdout.write(self.style.ERROR(str(type(e)) + ' ' + str(e)))
            return

        if len(users) > 0:
            self.stdout.write(self.style.SUCCESS(f'Successfully sent {len(users)} rating reminders.'))
        else:
            self.stdout.write(self.style.SUCCESS(f'No notifications were sent because there is currently no need for them.'))
        return
