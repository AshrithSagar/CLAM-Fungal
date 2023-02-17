#!/bin/bash

exp=$1

result_dir="experiments/results/"
outputdir="experiments/heatmaps/"

outputfile=${file}.zip
contents=${result_dir}

zip ${outputfile} ${contents}
