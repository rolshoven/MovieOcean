# About MovieOcean

This is the code for MovieOcean, available at https://movieocean.herokuapp.com. MovieOcean is a movie recommender system. In this system, we use a new approach that takes the personality into account. For the recommendations, we are using an adapted user-based collaborative filtering algorithm. We build neighborhoods according to the personalities of the users. For assessing the personalities, we are using the Five-Factor Model: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. For short: OCEAN. The findings of this system are only of academic interest. 


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

If you want activate the environment in your shell, use `pipenv shell`.

Additionally, you will need a file named `.env` and place it in the folder `MovieOCEAN`. There is already an example called `.env.example`. You will need these exact variables. For this, you will have to request a TMDB API-key, more information can be found at https://www.themoviedb.org/documentation/api.

## Troubleshooting

You might run into an error that links you to this site: https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html. For me, the problem was solved by using another version of Numpy: `pip install numpy==1.19.3`.

