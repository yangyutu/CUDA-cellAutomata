#!/usr/bin/env bash

DIR="logfiles"
if [ ! -e $DIR ]; then
	mkdir $DIR
fi

#----------------- gpu: global ------------------
nvcc cell_gpu.cu -o cell_gpu
PATH="$DIR/gpu_global.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..12}; do
	for i in {1..10}; do 
		timeval=$(./cell_gpu $[2**$dim] 1024)
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done

