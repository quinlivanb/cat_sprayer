#!/bin/bash

source /home/pi/PythonProjects/cat_sprayer/sprayer/bin/activate
. /home/pi/PythonProjects/cat_sprayer/env_setup
nohup python /home/pi/PythonProjects/cat_sprayer/cat_sprayer.py &
deactivate


