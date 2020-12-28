from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import CustomUser


class CustomUserAdmin(UserAdmin):
    readonly_fields = ['date_joined']

    model = CustomUser
    list_display = ('email', 'country', 'date_joined',)
    list_filter = ('date_joined', 'is_admin')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('date_of_birth', 'gender', 'country',)}),
        ('Permissions', {'fields': ('is_staff', 'is_admin')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_admin')}
         ),
    )
    search_fields = ('email',)
    ordering = ('-date_joined',)


admin.site.register(CustomUser, CustomUserAdmin)
