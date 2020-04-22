#!/bin/bash

source sprayer/bin/activate
. env_setup
nohup python cat_sprayer.py &
deactivate


