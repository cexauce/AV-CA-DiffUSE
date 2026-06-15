#!/bin/bash


clusters=("gruss") # don't try grele
seg_id=$1
total_seg=$2
gpus="1"
walltime="02:00:00"

full_command="oarsub -S \"./eval/SE_eval.sh ${seg_id} ${total_seg}\" -vv -l /nodes=1/gpu=${gpus},walltime=${walltime} -p \"cluster in (${clusters[@]})\" -q production -O ../OUT/oar_job.%jobid%.output -E ../OUT/oar_job.%jobid%.error"


echo $full_command
eval $full_command
