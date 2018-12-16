#!/bin/bash
#Number of files | Mean | Stddev
# 1281167 | 114667 | 338.58972223031224194536
mean=114667,24;
media=0
amountOfFiles=0;
R=$((RANDOM%1000+1));
stddev=0;
acc=0;
N=12;
outputpath="/home/breno/Desktop/train";

cd train/;
for i in `ls`;
        do cd $i;
        for j in `ls`;
                do std=$(stat -c '%s' "$j");
                aux=$((std-(mean)));
                acc=$(((acc)+((aux)^2)));
                amountOfFiles=$((amountOfFiles+1));
		media=$((media+(std)));
                done;
        cd ..; 
        done;
echo 'end for'
acc=$((acc / (amountOfFiles)));
stddev=$(echo "sqrt ( ($acc) )" | bc -l) ; 
echo 'stddev'
echo $stddev
echo 'amount of files'
echo $amountOfFiles
echo 'media final'
echo $((media/(amountOfFiles)))
