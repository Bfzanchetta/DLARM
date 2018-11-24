mean=114667,24;
amountOfFiles=0;
acc=0;
cd train/;
for i in `ls`;
        do cd $i;
        for j in `ls`;
                do std=$(stat -c '%s' "$j");
                aux=$(($std-($mean)));
                acc=$((($acc)+(($aux)^2)));
                amountOfFiles=$(($amountOfFiles+1));
                done;
        cd ..; 
        done;
echo 'end for'
acc=$(($acc / ($amountOfFiles)));
stddev=$(echo "sqrt ( ($acc) )" | bc -l) ; 
echo 'stddev'
echo $stddev

lowerICNinetyFive=$((($mean)-1,64*(($stddev) / ($amountOfFiles))));
upperICNinetyFive=$((($mean)+1,64*(($stddev) / ($amountOfFiles))));

lowerICNinetyNine=$((($mean)-2,965*(($stddev) / ($amountOfFiles))));
upperICNinetyNine=$((($mean)+2,965*(($stddev) / ($amountOfFiles))));

