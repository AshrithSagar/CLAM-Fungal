#!/bin/bash

config_dir="experiments/config/"

for configFile in "${config_dir}"*.yaml; do
    if [[ "${configFile}" != *"ignore"* ]]; then
        echo "Running main.py with ${configFile}"
        time python3 main.py --config ${configFile} && date
        mv ${configFile} "${config_dir}completed/${configFile##*/}"
    fi
done
