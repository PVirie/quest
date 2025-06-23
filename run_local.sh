#!/bin/bash

profile=$1

source .venv/bin/activate

shift 1
python3 $@ 
