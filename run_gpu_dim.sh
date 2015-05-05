#!/usr/bin/env bash

DIR="logfiles"
if [ ! -e $DIR ]; then
	mkdir $DIR
fi

#----------------- gpu: global ------------------
nvcc cell_gpu_dim.cu -o cell_gpu_dim
PATH="$DIR/gpu_dim.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..12}; do
	for i in {1..10}; do 
		timeval=$(./cell_gpu_dim $[2**$dim] 1024 1)
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done

#------------------ gpu: tex1d ------------------
# nvcc cell_gpu_tex1d.cu -o cell_gpu_tex1d
# PATH="$DIR/gpu_tex1d.log"
# if [ -e $PATH ]; then
# 	echo "File $PATH already exists, please delete or rename it first!" >&2
# 	exit 1
# fi
# 
# for dim in {6..12}; do
# 	for freq in {0..10}; do
# 		timeval=$(./cell_gpu_tex1d $[2**$dim] 1024 $[2**freq])
# 		echo $timeval >> $PATH
# 	done
# 	printf "\n" >> $PATH
# done
# 
# #------------------ gpu: tex2d ------------------
# nvcc cell_gpu_tex2d.cu -o cell_gpu_tex2d
# PATH="$DIR/gpu_tex2d.log"
# if [ -e $PATH ]; then
# 	echo "File $PATH already exists, please delete or rename it first!" >&2
# 	exit 1
# fi
# 
# for dim in {6..12}; do
# 	for freq in {0..10}; do
# 		timeval=$(./cell_gpu_tex2d $[2**$dim] 1024 $[2**freq])
# 		echo $timeval >> $PATH
# 	done
# 	printf "\n" >> $PATH
# done
