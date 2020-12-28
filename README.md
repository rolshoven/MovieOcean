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

