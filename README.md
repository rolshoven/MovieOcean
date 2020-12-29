# About MovieOcean

## General

This is the code for MovieOcean, available at https://movieocean.herokuapp.com. MovieOcean is a movie recommender system. In this system, we use a new approach that takes the personality into account. For the recommendations, we are using an adapted user-based collaborative filtering algorithm. We build neighborhoods according to the personalities of the users. For assessing the personalities, we are using the Five-Factor Model: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. For short: OCEAN. The findings of this system are only of academic interest, and we randomized all data so that there is no way to trace information back to users. This system was made by **Luca Rolshoven** and **Corina Masanti**.

## Structure

We structured our project in folders:
- MovieOCEAN: Contains general settings.
- accounts: Contains everything that concerns the user accounts. For example, creating a new user, calculating the neighborhood, and so on.
- data: Contains the questions we used to assess the personality, the template of the rating reminder (is sent one week after a new account is made), personality stereotypes for the genre (taken from the findings of Cantador, I., Fernández-Tobías, I., & Bellogín, A. (2013). Relating personality types with user preferences in multiple entertainment domains. In CEUR workshop proceedings. Shlomo Berkovsky.), and a list of countries the user can select when registering.
- evaluation: Contains the code we used to evaluate our system (comparison of standard user-based CF with our approach). We used the RMSE to evaluate the accuracy of the estimated ratings and the F1-score to evaluate the relevance of our recommendations.
- movie: Contains everything that concerns the movies. For example, displaying the movies, creating the watchlist, and so on.
- notification: Contains logic for sending the rating reminder.
- questionnaire: Contains logic of the personality questionnaire.

For further information, please have a look at the documentation within the files.


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

