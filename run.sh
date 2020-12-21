#!/bin/bash
echo $HOME
echo $PATH

export PYENV_ROOT="/home/otto-chriek/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate bias-env
python predict.py -A $1 -O $3/test.preds.txt
