#!/bin/bash
good_clusters_HDD=("gruss") # "grele", "grue", "gruss"    "graffiti", "grele", "grue"
good_clusters_SSD=("gruss")
exotic_SSD="grouille"
name=$1
gpus="2"
walltime="10:00:00"

full_command="oarsub  -S ./runs/infoNCE_trainable_audio_enc/av_diffse_late_fusion_cogenav_icp52_6M_warmup_100_beta0_3000_beta_t_step_unweighted.sh -vv -l /nodes=1/gpu=${gpus},walltime=${walltime} -p \"cluster in (${good_clusters_HDD[@]})\" -q production -n ${name} -O ../OUT/oar_job.%jobid%_${name}.output -E ../OUT/oar_job.%jobid%_${name}.error"
# full_command="oarsub  -S ./train.sh -vv -l /nodes=1/gpu=${gpus},walltime=${walltime} -p \"cluster in (${exotic_SSD[@]})\" -t exotic -n ${name} -O ../OUT/oar_job.%jobid%.output -E ../OUT/oar_job.%jobid%.error"  ../OUT/oar_job.%jobid%_${name}.output -E ../OUT/oar_job.%jobid%_${name}.error

echo $full_command

read -p "Continue and launch this command? (y/n)" 

case $REPLY in 
        [Yy]* ) 
        eval $full_command
        echo "Command launched"
        ;;
        "" )
        eval $full_command
        echo "Command launched"
        ;;

        * ) 
        echo "Aborted"
        exit
        ;;
esac