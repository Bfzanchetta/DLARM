#!/bin/bash
#Number of files | Mean | Stddev
# 1281167 | 114667 | 338.58972223031224194536
R=$((RANDOM%1000+1));
N=12;
outputpath="/home/nvidia/dataset/train";
soma=0;
mean=114667.24;
stddev=338.58972223031224194536;
limit1=60400.75;
limit3=147924.00;
totaldataset=0.0;

zona1=0;
zona2=0;
zona3=0;
zona4=0;

balance=0.0;
#for i in `ls`; do
#        mkdir(outputpath/$i);
#        done;
#localMedian=0;
#localStddev=0;

cd $outputpath;

for i in `ls`; do
	cd $i;
	a=( * )
	randf=()
	fSize=0;
	for((i=0;i<N && ${#a[@]};++i)); do
		((j=RANDOM%${#a[@]}))
		fSize=$(stat -c '%s' "${a[j]}");
		if [ `echo $fSize'<'$limit1 | bc -l` == 1 ]; then
			echo 'arquivo pequeno'
			zona1=$((zona1+1));
			totaldataset=`echo $totaldataset + $fSize | bc -l`;
		elif [ `echo $fSize'>='$limit1 | bc -l` == 1 -a `echo $fSize'<'$mean | bc -l` == 1 ]; then
            		soma=`echo "$mean - $fSize" | bc`;
			balance=`echo "$balance + $soma" | bc`;
			zona2=$((zona2+1));
			echo 'arquivo 1'
			totaldataset=`echo $totaldataset + $fSize | bc -l`;
        	elif [ `echo $fSize'>'$mean | bc -l` == 1 -a `echo $fSize'<='$limit3 | bc -l` == 1 ]; then
			echo 'arquivo 3'
			zona3=$((zona3+1));
			totaldataset=`echo $totaldataset + $fSize | bc -l`;
			if [ `echo $balance'>'$fSize | bc -l` == 1 ]; then
				soma=`echo "$fSize - $mean" | bc`;
                		balance=`echo "$balance + $soma" | bc`;
            		else
                		#draw again
				echo 'oi'
            		fi
		elif [ `echo $fSize'>'$limi3 | bc -l` == 1 ]; then
			echo 'arquivo grande'
			zona4=$((zona4+1));
			totaldataset=`echo $totaldataset + $fSize | bc -l`;
        	else
	    		echo '**************'
   	    		echo '     Erro     '
	    		echo '**************'
        	fi
        	done;
	cd ..;
    	done;
echo 'Resultados'
echo 'zona1'
echo $zona1
echo 'zona2'
echo $zona2
echo 'zona3'
echo $zona3
echo 'zona4'
echo $zona4
