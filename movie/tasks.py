import functools

from accounts.models import CustomUser
from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.db.utils import IntegrityError

from .models import Recommendation
import time


logger = get_task_logger(__name__)

CACHE_LOCK_EXPIRE = 30


def no_simultaneous_execution(f):
    """
    Decorator that prevents a task form being executed with the
    same *args and **kwargs more than one at a time.
    Credits: Andreas Bergström (https://stackoverflow.com/a/59301422)
    """
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Create lock_id used as cache key
        lock_id = kwargs['user_id']

        # Timeout with a small diff, so we'll leave the lock delete
        # to the cache if it's close to being auto-removed/expired
        timeout_at = time.monotonic() + CACHE_LOCK_EXPIRE - 3

        # Try to acquire a lock, or put task back on queue
        lock_acquired = cache.add(lock_id, True, CACHE_LOCK_EXPIRE)
        if not lock_acquired:
            self.apply_async(args=args, kwargs=kwargs, countdown=3)
            return

        try:
            f(self, *args, **kwargs)
        finally:
            # Release the lock
            if time.monotonic() < timeout_at:
                cache.delete(lock_id)
    return wrapper


@shared_task(bind=True)
@no_simultaneous_execution
def recompute_recommendations(self, user_id, delay, max_recommendations):
    """
    See https://docs.celeryproject.org/en/latest/tutorials/task-cookbook.html#ensuring-a-task-is-only-executed-one-at-a-time.
    """
    try:
        user = CustomUser.objects.get(id=user_id)
        Recommendation.update_old(user, delay, max_recommendations)
        return True
    except IntegrityError:
        logger.info(f'Recomputation of recommendations for user {user_id} already succeeded.')
