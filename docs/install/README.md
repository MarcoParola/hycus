# Install the project

## Windows

Run the following code snippet tocCreate the virtualenv (you can also use conda) and install the dependencies of `requirements.txt`

```sh
python3 -m venv env
env\Scripts\activate
python -m pip install -r requirements.txt
mkdir data
```

## Linux

```sh
python3 -m venv env
. env/bin/activate
python -m pip install -r requirements.txt
mkdir data
```