#!/bin/bash
echo W:myRank is $SLURM_PROCID
IMG=$1
CMD=$2
outPath=$3

if [ $SLURM_PROCID -eq 0 ] ; then 
   echo W:IMG=$IMG 
   echo W:CMD=$CMD
   #echo Q:fire $
fi

podman-hpc run -it \
	   --volume $outPath:/wrk \
	   --workdir /wrk \
	   $IMG $CMD --myRank $SLURM_PROCID
