#!/bin/bash
declare -a models;
declare -a configuration;
declare -a repetition;

models=(MX-GPU-ALEX-mImgNet10 MX-GPU-CNN-mImgNet10 MX-GPU-SQZ-mImgNet10 MX-GPU-DNS-mImgNet10 MX-GPU-VGG-mImgNet10 MX-GPU-GGL-mImgNet10 MX-GPU-RES-mImgNet10 MX-CPU-ALEX-mImgNet10 MX-CPU-CNN-mImgNet10 MX-CPU-SQZ-mImgNet10 MX-CPU-DNS-mImgNet10 MX-CPU-VGG-mImgNet10 MX-CPU-GGL-mImgNet10 MX-CPU-RES-mImgNet10 PT-GPU-ALEX-mIMGN10 PT-GPU-DNS-mIMGN10 PT-GPU-GGL-mIMGN10 PT-GPU-INC-mIMGN10 PT-GPU-RES-mIMGN10 PT-GPU-SQZ-mIMGN10 PT-GPU-VGG-mIMGN10 PT-CPU-ALEX-mIMGN10 PT-CPU-DNS-mIMGN10 PT-CPU-GGL-mIMGN10 PT-CPU-INC-mIMGN10 PT-CPU-RES-mIMGN10 PT-CPU-SQZ-mIMGN10 PT-CPU-VGG-mIMGN10);
configuration=(3);
repetition=(1);

for i in ${repetition[@]}; do
	for j in ${configuration[@]}; do
		for k in ${models[@]}; do
			echo $k'_CFG'$j'_'$i >> experimentos.txt
		done;
	done;
done;


