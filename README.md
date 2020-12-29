# About MovieOcean

## General

This is the code for MovieOcean, available at https://movieocean.herokuapp.com. MovieOcean is a movie recommender system. In this system, we use a new approach that takes the personality into account. For the recommendations, we are using an adapted user-based collaborative filtering algorithm. We build neighborhoods according to the personalities of the users. For assessing the personalities, we are using the Five-Factor Model: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. For short: OCEAN. The findings of this system are only of academic interest. This system was made by **Luca Rolshoven** and **Corina Masanti** in the course of a seminar at the University of Fribourg (CH).

## Structure

We structured our project in folders:
- MovieOCEAN: Contains general settings.
- accounts: Contains everything that concerns the user accounts. For example, creating a new user, calculating the neighborhood, and so on.
- data: Contains the questions we used to assess the personality, the template of the rating reminder (is sent one week after a new account is made), personality stereotypes for the genre (taken from the findings of Cantador, Iván, Ignacio Fernández-Tobías, and Alejandro Bellogín. "Relating personality types with user preferences in multiple entertainment domains." CEUR workshop proceedings. Shlomo Berkovsky, 2013.), and a list of countries the user can select when registering.
- evaluation: Contains the code we used to evaluate our system (comparison of standard user-based CF with our approach). We used the RMSE to evaluate the accuracy of the estimated ratings and the F1-score to evaluate the relevance of our recommendations.
- movie: Contains everything that concerns the movies. For example, displaying the movies, creating the watchlist, and so on.
- notification: Contains logic for sending the rating reminder.
- questionnaire: Contains logic of the personality questionnaire.

For further information, please have a look at the documentation within the files.

## Credits

For our system, we are using the movie database TMDB (https://www.themoviedb.org/) together  with  the  wrapper  library  tmdbsimple (https://github.com/celiao/tmdbsimple). In addition, the questions for the personality questionnaire were taken from the International Personality Item Pool (https://ipip.ori.org/). The information about the big five personality traits was mostly taken from Roccas, Sonia, et al. "The big five personality factors and personal values." Personality and social psychology bulletin 28.6 (2002): 789-801. Available at https://journals.sagepub.com/doi/abs/10.1177/0146167202289008.

Additionally, we are using the following:
- Bulma - CSS framework: https://bulma.io/
- Pandas - data analysis and manipulation tool, built on top of the python programming language: https://pandas.pydata.org/
- Celery - distributed task queue: https://pandas.pydata.org/
- Sendgrid API: https://sendgrid.com/docs/API_Reference/index.html
- Heroku: https://www.heroku.com/


# How to install

## Prerequisites

You will have to install `pipenv` if you have not done so already. You can install `pipenv` running

```shell
pip install pipenv
```

Afterwards, use the following command to install the dependencies:

```shell
pipenv install
```

If you want to add another dependency, which is not yet included, just install it using pipenv:

```shell
pipenv install some_random_package
```

If you only use the dependency during development, you can add the `--dev` flag:

```shell
pipenv install some_random_package --dev
```

If you want to activate the environment in your shell, use `pipenv shell`.

Additionally, you will need a file named `.env` and place it in the folder `MovieOCEAN`. There is already an example called `.env.example`. You will need these exact variables. For this, amongst other things, you will have to request a TMDB API-key, more information can be found at https://www.themoviedb.org/documentation/api.

## Troubleshooting

You might run into an error that links you to this site: https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html. For us, the problem was solved by using another version of Numpy with this command: `pip install numpy==1.19.3`.

