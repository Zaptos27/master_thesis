#!/bin/bash

python -m venv .venv
pip install tensorflow[and-cuda]==2.15.* apache_beam tensorflow_datasets
source .venv/bin/activate