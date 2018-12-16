#!/bin/bash
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

#lowerICNinetyFive=$((($mean)-1,64*(($stddev) / ($amountOfFiles))));
#upperICNinetyFive=$((($mean)+1,64*(($stddev) / ($amountOfFiles))));

#lowerICNinetyNine=$((($mean)-2,965*(($stddev) / ($amountOfFiles))));
#upperICNinetyNine=$((($mean)+2,965*(($stddev) / ($amountOfFiles))));

#Selects the random folders#


for k in `ls`; do
	cd $k;
	a=( * )
	randf=()
	temp=0;
	for((i=0;i<N && ${#a[@]};++i)); do
    		((j=RANDOM%${#a[@]}))
    		echo ${a[j]}
		temp=$(stat -c '%s' "${a[j]}");
		echo 'temp'
		echo $temp
		if [ $temp -gt 97503 ]; then
			echo 'ulala'
		else
			echo 'notnotnotnotnotnot'
		fi
		done;
	cd ..
	done;
echo 'fim do loop'	
    #resto da logica de verificacao aqui
    
#for i in `ls`; do
        #pegar a folder com o indice
#        mkdir(outputpath/$i);
#        done;

localMedian=0;
localStddev=0;
balance=0;

#for i in `ls`; do
#        cd $i;
#        a=( * )
#        randf=()
#        for((i=0;i<N && ${#a[@]};++i)); do
#                ((j=RANDOM%${#a[@]}))
#                if [ ${a[j]} -get $($(mean)-$(2*($(stddev)))) -a ${a[j]} -lt $($(mean)-$(stddev)) ]; then
#                        $balance=$(balance+(mean-a[j]));
#                elif [ ${a[j]} -get $(mean-stddev) -a ${a[j]} -lt $(mean) ]; then
#                        $balance=$(balance+(mean-a[j]));
#                elif [ ${a[j]} -get $(mean) -a ${a[j]} -lt $(mean+stddev) ]; then
#                        if [ $balance -get  ${a[j]} ]; then
#                                $balance=$(balance-(mean-a[j]));
#                        else
#                                #draw again
#				echo 'oi'
#                        fi
#                elif [ ${a[j]} -get $(mean+stddev) -a ${a[j]} -lt $(mean+(2*stddev)) ]; then
#                        if [ $balance -get  ${a[j]} ]; then
#                                $balance=$(balance-(mean-a[j]));
#                        else
#                                #draw again
#				echo 'draw again'
#                        fi
#                else
#   			echo 'oi'
#                fi
#                done;
#        cd ..;
#        done;
