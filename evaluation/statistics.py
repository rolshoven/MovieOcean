"""
This file synchronizes the data from the database and stores it in csv files for further
evaluation. The user ids and movie ids will be randomized so that there is no way to
establish a connection between certain users and certain movies/ratings.
"""

# Set up django to work in this stand-alone script
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MovieOCEAN.settings')
django.setup()

# Other imports
from collections import Counter
from datetime import date
import matplotlib.pyplot as plt
from accounts.models import CustomUser


def abs_val(pct):
    return '{}'.format(int(round(pct * len(users) / 100.0)))


# Get statistics of the users
users = CustomUser.objects.all()
countries = Counter([u.get_country_display() for u in users])
ages = Counter([(date.today() - u.date_of_birth).days // 365 for u in users])
genders = Counter([u.get_gender_display() for u in users])

# Plot country distribution
c = list(countries.keys())
c[c.index('United States of America (the)')] = 'USA'
plt.bar(c, countries.values(), color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(c, rotation=90)
plt.margins(0.1)
plt.subplots_adjust(bottom=0.25)
for i, v in enumerate(countries.values()):
    x_shift = - 0.15 if v > 9 else -0.08
    plt.text(i + x_shift, v + 1, str(v), color='lightsteelblue', fontweight='bold')
plt.yticks([])
plt.savefig(os.path.join('plots', 'statistics', 'countries.jpg'))
plt.clf()

# Plot age distribution
plt.bar(ages.keys(), ages.values(), color=(0.2, 0.4, 0.6, 0.6))
plt.margins(0.1)
x_range = range(min(ages), max(ages) + 6 - max(ages) % 5)
plt.xticks(x_range, [i if i % 5 == 0 else '' for i in x_range])
plt.yticks(range(max(ages.values()) + 1))
plt.subplots_adjust(bottom=0.1)
plt.savefig(os.path.join('plots', 'statistics', 'ages.jpg'))
plt.clf()

# Plot gender distribution
colors = ['#54478c', '#2c699a', '#048ba8', '#0db39e', '#16db93', '#83e377', '#efea5a'][:len(genders)]
patches, _, _ = plt.pie(genders.values(), colors=colors, startangle=90, autopct=abs_val)
plt.legend(patches, genders.keys())
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'statistics', 'genders.jpg'))
plt.clf()

