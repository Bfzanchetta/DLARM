#!/bin/bash
# Author: Breno Fanchiotti Zanchetta
cd /home/nvidia/dataset/train   #Directory of the dataset you wish to reduce
total=0;
counter=1;
limit=6500000;   #Tells how much the new dataset folder must have of maximum length
totalamount=0;
outputpath="/home/nvidia/dataset/SmallerSet";
for i in `ls`;do
	cd $i;
	contador=0;
	perfolder=0;
	mkdir $outputpath/$i;
	for j in `ls -rS`; do                        #if one wishes to get greater images, just erase the -r and leave -S
		if [ `echo $contador'<'$limite | bc -l` == 1 ]; then
			fSize=$(stat -c '%s' "${j}");
			cp -ax $j $outputpath/$i/$j;
			contador=`echo "$contador + $fSize" | bc`;
			perfolder=$((perfolder+1));
			totalamount=`echo "$totalamount + $fSize" | bc`;			
		else
			break;
		fi;
		done;
	echo 'Na folder'
	echo $counter; 	
	echo $perfolder;
	counter=$((counter+1))
	total=$((total+perfolder));
	cd ..;	
	done;
echo 'counter'
echo $counter
echo 'total'
echo $total
echo 'total size'
echo $totalamount
totalamount=`echo "$totalamount / $counter" | bc`;
echo 'average between all new folders'
echo $totalamount

echo 'fim'
