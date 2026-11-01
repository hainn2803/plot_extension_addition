#!/bin/bash
set -e

python3 addition_train_standard_grid.py --resume
python3 addition_train_gradual_grid.py --resume