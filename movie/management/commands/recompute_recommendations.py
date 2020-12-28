from django.core.management.base import BaseCommand, CommandError
from accounts.models import CustomUser
from movie.models import Recommendation
from django.conf import settings
import os

msg_path = os.path.join(settings.BASE_DIR, 'data', 'rating_reminder_message.txt')


class Command(BaseCommand):
    help = 'Recompute the recommendations of each user (new users first)'

    def add_arguments(self, parser):
        parser.add_argument('max_recommendations', nargs='?', type=int, default=100)
        parser.add_argument('new_users_only', nargs='?', type=bool, default=False)

    def handle(self, *args, **options):
        # try:
        users = CustomUser.objects.filter(has_personality_profile=True)
        rec_users = [u for u in users if u.recommendations.all().exists()]
        no_rec_users = [u for u in users if u not in rec_users]

        max_recs = options['max_recommendations']
        new_only = options['new_users_only']

        for u in no_rec_users:
            Recommendation.save_recommendations_for_user(u, max_recs)

        if not new_only:
            for u in rec_users:
                u.update_cached_recommendations(max_recs)

        # except Exception as e:
        #     self.stdout.write(self.style.ERROR(str(type(e)) + ' ' + str(e)))
        #     return

        if new_only:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully updated recommendations for {len(no_rec_users)} new users.'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Successfully updated recommendations for {len(users)} users.'))
        return


