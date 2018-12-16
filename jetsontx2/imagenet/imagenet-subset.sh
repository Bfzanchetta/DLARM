#!/bin/bash
#Number of files | Mean | Stddev
# 1281167 | 114667 | 338.58972223031224194536
mean=114667,24;
R=$((RANDOM%1000+1));
stddev=338.58972223031224194536
N=12;
outputpath="/home/breno/Desktop/train";
balance=0;

fourthLimit=115344,419444461;
thirdLimit=115005,82972223;
secondLimit=114328,65027777;
firstLimit=113990,060555539;

#lowerICNinetyFive=$((($mean)-1,64*(($stddev) / ($amountOfFiles))));
#upperICNinetyFive=$((($mean)+1,64*(($stddev) / ($amountOfFiles))));
#lowerICNinetyNine=$((($mean)-2,965*(($stddev) / ($amountOfFiles))));
#upperICNinetyNine=$((($mean)+2,965*(($stddev) / ($amountOfFiles))));
#Selects the random folders#
    
#for i in `ls`; do
        #pegar a folder com o indice
#        mkdir(outputpath/$i);
#        done;

#localMedian=0;
#localStddev=0;


for i in `ls`; do
        cd $i;
        a=( * )
        randf=()
	fileSize=0;
        for((i=0;i<N && ${#a[@]};++i)); do
                ((j=RANDOM%${#a[@]}))
		fileSize=$(stat -c '%s' "${a[j]}");
		var=$(awk 'BEGIN{ print "'$fileSize'">="'$firstLimit'" }');
		var1=$(awk 'BEGIN{ print "'$fileSize'">="'$secondLimit'" }');
		var2=$(awk 'BEGIN{ print "'$fileSize'">="'$mean'" }');
		var3=$(awk 'BEGIN{ print "'$fileSize'">="'$thirdLimit'" }');
		var4=$(awk 'BEGIN{ print "'$fileSize'">="'$fourthLimit'" }');
		var5=$(awk 'BEGIN{ print "'$fileSize'"<"'$firstLimit'" }');
		var6=$(awk 'BEGIN{ print "'$fileSize'"<"'$secondLimit'" }');
		var7=$(awk 'BEGIN{ print "'$fileSize'"<"'$mean'" }');
		var8=$(awk 'BEGIN{ print "'$fileSize'"<"'$thirdLimit'" }');
		var9=$(awk 'BEGIN{ print "'$fileSize'"<"'$fourthLimit'" }');
		
                if [ "$var" -eq 1 -a "$var6" -eq 1 ]; then
                        $balance=$(balance+(mean-fileSize));
                elif [ "$var1" -eq 1 -a "$var7" -eq 1 ]; then
                        $balance=$(balance+(mean-fileSize));
                elif [ "$var2" -eq 1 -a "$var8" -eq 1 ]; then
                        if [ $balance -get  $fileSize ]; then
                                $balance=$(balance-(mean-fileSize));
                        else
                                #draw again
				echo 'oi'
                        fi
                elif [ "$var3" -eq 1 -a "$var9" -eq 1 ]; then
                        if [ $balance -get  $fileSize ]; then
                                $balance=$(balance-(mean-fileSize));
                        else
                                #draw again
				echo 'draw again'
                        fi
                else
   			echo 'oi'
                fi
                done;
        cd ..;
        done;


key1=11.3
result=12.5
var=$(awk 'BEGIN{ print "'$key1'"<"'$result'" }')    
# or var=$(awk -v key=$key1 -v result=$result 'BEGIN{print result<key?1:0}')
# or var=$(awk 'BEGIN{print "'$result'"<"'$key1'"?1:0}')
# or 
if [ "$var" -eq 1 -a 3 -lt 5 ];then
  echo "do something"
else
  echo "result more than key"
fi
