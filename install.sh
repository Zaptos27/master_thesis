#!/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install tensorflow[and-cuda]==2.15.* apache_beam tensorflow_datasets
