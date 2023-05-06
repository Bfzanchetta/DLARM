#!/bin/bash
#input=parametros.txt
#Insert permutable values of each training variable in the three variables below
declare -a numero=("5" "10") 
declare -a batch=("4")
declare -a learning=("0.1" "0.01" "0.001" "0.0001" "0.5" "0.05" "0.005" "0.0005") 
declare -a smooth=("0.1" "0.01" "0.001" "0.5" "0.05" "0.005") 
for i in "${numero[@]}"; do
	for j in "${batch[@]}"; do
		for k in "${learning[@]}"; do
			for l in "${smooth[@]}"; do
			echo $i $j $k $l >> arquivo.txt
			done;		
		done;
	done;
done;

mes=`date | cut -d' ' -f2`
dia=`date | cut -d' ' -f3`
hora=`date | cut -d' ' -f4`
design="design_"$mes"_"$dia"_"$hora".txt"
sort -R arquivo.txt > "design_"$mes"_"$dia"_"$hora".txt"
input=$design

while read line
do
        num=$(echo $line | cut -d' ' -f1)
        bat=$(echo $line | cut -d' ' -f2)
        lea=$(echo $line | cut -d' ' -f3)
        smo=$(echo $line | cut -d' ' -f4)
	#python Alex-mImgNet10.py --n $num --b $bat --l $lea --s $smo
	echo $num $bat $lea $smo
done < "$input"
