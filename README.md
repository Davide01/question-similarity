# Question similarity classifier

## Setup
You will need python 3.14 and poetry to setup this project. I suggest you have a pyenv install for easier python version management. As a prerequisite you need to have poetry installed.

Steps:
1. If you don't have already install python 3.14 with pyenv:
    pyenv install 3.14
2. Setup as local default env:
    pyenv local 3.14
3. Let poetry use the env
    poetry env use 3.14
4. Install dependencies
    poetry sync

## Data cleaning
You can perform separately the cleaning of the data by running the script


## Docker
docker build -t question-sim-api .

docker run --rm -p 8000:8000 question-sim-api
