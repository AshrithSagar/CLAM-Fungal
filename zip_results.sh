#!/bin/bash

# Usage
# ./zip_results.sh exp_00_s1 splits_0 filename_for_zip

exp=$1
split=$2
file=$3

result_dir="experiments/results/"
outputdir="experiments/heatmaps/"

outputfile="${outputdir}${file}.zip"
contents="${result_dir}${exp}/${split}/heatmaps/*"

zip ${outputfile} ${contents}
