#!/bin/bash
#Number of files | Mean | Stddev
# 1281167 | 114667 | 338.58972223031224194536
mean=114667,24;
media=0
amountOfFiles=0;
stddev=0;
acc=0;
#std=0;
outputpath="/home/breno/Desktop/train";

cd train/;
for i in `ls`;
        do cd $i;
        for j in `ls`;
                do std=$(stat -c '%s' "$j");
		aux=`echo "$std - $mean" | bc`;
		aux=`echo "$aux * $aux" | bc`;
		acc=`echo "$acc + $aux" | bc`;
                acc=$(((acc)+((aux)^2)));
                amountOfFiles=$((amountOfFiles+1));
		media=$((media+(std)));
                done;
        cd ..; 
        done;
echo 'end for'

#acc=$((acc / (amountOfFiles)));
acc=`echo "$acc / $amountOfFiles" | bc`;

stddev=$(echo "sqrt ( ($acc) )" | bc -l) ; 

echo 'Total de Arquivos'
echo $amountOfFiles
echo 'Media do Dataset'
echo $acc
echo 'Stddev'
echo $stddev
