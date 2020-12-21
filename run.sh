#!/bin/bash

pyenv activate bias-env
python predict.py -A $1 -O $3/test.preds.txt
