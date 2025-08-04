#!/bin/bash

profile=$1

source secrets.env
source .venv/bin/activate

shift 1
python3 $@ 
