#!/usr/bin/env bash

DIR="logfiles"
if [ ! -e $DIR ]; then
	mkdir $DIR
fi

#--------------------- cpu ----------------------
g++ cell_cpu.cpp -o cell_cpu
PATH="$DIR/cpu.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..7}; do
	for freq in {0..10}; do
		timeval=$(./cell_cpu $[2**$dim] 1024 $[2**freq])
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done

#----------------- gpu: global ------------------
nvcc cell_gpu.cu -o cell_gpu
PATH="$DIR/gpu_global.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..12}; do
	for freq in {0..10}; do
		timeval=$(./cell_gpu $[2**$dim] 1024 $[2**freq])
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done

#------------------ gpu: tex1d ------------------
nvcc cell_gpu_tex1d.cu -o cell_gpu_tex1d
PATH="$DIR/gpu_tex1d.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..12}; do
	for freq in {0..10}; do
		timeval=$(./cell_gpu_tex1d $[2**$dim] 1024 $[2**freq])
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done

#------------------ gpu: tex2d ------------------
nvcc cell_gpu_tex2d.cu -o cell_gpu_tex2d
PATH="$DIR/gpu_tex2d.log"
if [ -e $PATH ]; then
	echo "File $PATH already exists, please delete or rename it first!" >&2
	exit 1
fi

for dim in {6..12}; do
	for freq in {0..10}; do
		timeval=$(./cell_gpu_tex2d $[2**$dim] 1024 $[2**freq])
		echo $timeval >> $PATH
	done
	printf "\n" >> $PATH
done
