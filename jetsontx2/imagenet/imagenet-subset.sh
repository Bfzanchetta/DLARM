#!/bin/bash
mean=114667,24;
amountOfFiles=0;
R=$((RANDOM%1000+1));
stddev=0;
acc=0;
N=12;
outputpath="/home/nvidia/Desktop/folder";

cd train/;
for i in `ls`;
        do cd $i;
        for j in `ls`;
                do std=$(stat -c '%s' "$j");
                aux=$((std-(mean)));
                acc=$(((acc)+((aux)^2)));
                amountOfFiles=$((amountOfFiles+1));
                done;
        cd ..; 
        done;
echo 'end for'
acc=$((acc / (amountOfFiles)));
stddev=$(echo "sqrt ( ($acc) )" | bc -l) ; 
echo 'stddev'
echo $stddev

#lowerICNinetyFive=$((($mean)-1,64*(($stddev) / ($amountOfFiles))));
#upperICNinetyFive=$((($mean)+1,64*(($stddev) / ($amountOfFiles))));

#lowerICNinetyNine=$((($mean)-2,965*(($stddev) / ($amountOfFiles))));
#upperICNinetyNine=$((($mean)+2,965*(($stddev) / ($amountOfFiles))));

#Selects the random folders#
a=( * )
randf=()
for((i=0;i<N && ${#a[@]};++i)); do
    ((j=RANDOM%${#a[@]}))
    
    echo ${a[j]}
    #resto da logica de verificacao aqui
    
for i in `ls`; do
        #pegar a folder com o indice
        mkdir(outputpath/$i);
        done;

localMedian=0;
localStddev=0;

for i in `ls`; do
        cd $i;
        a=( * )
        randf=()
        for((i=0;i<N && ${#a[@]};++i)); do
                ((j=RANDOM%${#a[@]}))
                if [ ${a[j]} -get $(mean-(2*stddev)) -a ${a[j]} -lt $(mean-stddev) ]
                then
                        
                elif [  ${a[j]} -get $(mean-stddev) -a ${a[j]} -lt $(mean)  ]
                then
                
                elif [ ${a[j]} -get $(mean) -a ${a[j]} -lt $(mean+stddev) ]
                then
                
                elif [ ${a[j]} -get $(mean+stddev) -a ${a[j]} -lt $(mean+(2*stddev)) ]
                then
                
                else
   
                fi
                done;
        cd ..;
        done;

                
