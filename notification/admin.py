from django.contrib import admin
from .models import Notification


class NotificationAdmin(admin.ModelAdmin):
    model = Notification

    list_display = ('sent_to', 'type', 'date')
    list_filter = ('type', 'date',)
    search_fields = ('sent_to__email', 'type',)
    ordering = ('-date',)


# Register your models here.
admin.site.register(Notification, NotificationAdmin)
