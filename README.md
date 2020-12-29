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

Additionally, you will need a file named `.env` and place it in the folder `MovieOCEAN`. There is already an example called `.env.example`. You will need these exact variables. For these, you will need to request a API-key of TMDB.

