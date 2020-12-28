from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views import generic, View
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.forms import DateInput
from django import forms
import threading
from django.views.generic import TemplateView

from.models import CustomUser


class UserCreationForm(forms.ModelForm):
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Password confirmation', widget=forms.PasswordInput)

    class Meta:
        model = CustomUser
        fields = ('email', 'date_of_birth', 'gender', 'country')
        widgets = {
            'date_of_birth': DateInput(attrs={'type': 'date'})
        }

    def clean_password2(self):
        # Check that the two password entries match
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError('Password mismatch')
        return password2

    def save(self, commit=True):
        user = super(UserCreationForm, self).save(commit=False)
        user.set_password(self.cleaned_data['password1'])
        user.date_of_birth = self.cleaned_data.get('date_of_birth')
        user.gender = self.cleaned_data.get('gender')
        user.country = self.cleaned_data.get('country')
        if commit:
            user.save()
        return user


class SignUpView(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'registration/signup.html'


class AccountsView(LoginRequiredMixin, View):
    template_name = 'accounts/account.html'

    def get(self, request):
        return render(request, self.template_name, {'user': request.user})


class HelpView(LoginRequiredMixin, TemplateView):
    template_name = 'help/help.html'


@login_required
def password_change(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password was updated successfully!')
            return redirect('password_change')
        else:
            messages.error(request, 'Your password could not be updated. Please try again.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'registration/password_change.html', {
        'form': form
    })


def password_reset_done(request):
    messages.info(request, """
        If you entered an e-mail that is registered in our system, you will receive
        an e-mail with further instructions to reset your password. If you did not receive
        an e-mail, make sure you have entered the correct e-mail address and check your spam
        folder.
    """)
    return redirect('password_reset')


@login_required
def home(request):
    if request.user.has_personality_profile:
        return redirect('movie:recommendations')
    return render(request, 'registration/welcome_message.html')


def cookie_info(request):
    return render(request, 'registration/cookies.html')


@login_required
def delete_account(request):
    if request.user.has_personality_profile:
        # Trigger recomputation of neighborhoods
        t = threading.Thread(target=CustomUser.calculate_neighborhoods)
        t.setDaemon(False)  # Stop thread after it completion
        t.start()
    request.user.delete()
    return render(request, 'registration/account_deleted.html')


@login_required
def toggle_rating_reminders(request):
    request.user.rating_reminders = not request.user.rating_reminders
    request.user.save()
    messages.info(request, 'Rating reminders successfully deactivated!')
    return redirect('account')


