#!/bin/bash
path='/home/gppd/folder'
cd "$path";
for i in `ls *.csv`; do
	a=0;
	ram=0;
	cpu1=0;
	cpu2=0;
	cpu3=0;
	cpu4=0;
	cpu5=0;
	cpu6=0;
	gpu=0;
	f=0;
	vdd1=0;
	vdd2=0;
	vdd3=0;
	vdd4=0;
	vdd5=0;
	vdd6=0;
	vddtotal=0;
	nome=`echo "$i" | cut -d "." -f 1`
	echo "ram/7860,cpu0,cpu1,cpu2,cpu3,cpu4,cpu5,gpu,vddgpu,vddsoc,vddwifi,vddin,vddcpu,vddddr,vddtotal"
	while IFS= read -r line
		do
		a=`echo "$line" | cut -d "," -f 1`
		cpu2=`echo "$line" | cut -d "," -f 2`
		cpu3=`echo "$line" | cut -d "," -f 3`
		cpu4=`echo "$line" | cut -d "," -f 4`
		cpu5=`echo "$line" | cut -d "," -f 5`
		f=`echo "$line" | cut -d "," -f 6`
		cpu6=`echo "$f" | cut -d " " -f 1`
		cpu6=`echo "$cpu6" | cut -d "]" -f 1`
		gpu=`echo "$f" | cut -d " " -f 5`
		ram=`echo "$a" | cut -d " " -f 2`
		ram=`echo "$ram" | cut -d "/" -f 1`
		cpu1=`echo "$a" | cut -d "[" -f 2`
		vdd1=`echo "$f" | cut -d " " -f 15`
		vdd2=`echo "$f" | cut -d " " -f 17`
		vdd3=`echo "$f" | cut -d " " -f 19`
		vdd4=`echo "$f" | cut -d " " -f 21`
		vdd5=`echo "$f" | cut -d " " -f 23`
		vdd6=`echo "$f" | cut -d " " -f 25`
		##separate percentage of cpu1
                if [ $cpu1 == "off" ]; then
                        :
                else
                        cpu1=`echo "$cpu1" | cut -d "@" -f 1`
                fi;
                ##percentage of cpu2
                if [ $cpu2 == "off" ]; then
                        :
                else
                        cpu2=`echo "$cpu2" | cut -d "@" -f 1`
                fi;
                ##cpu3
                if [ $cpu3 == "off" ]; then
                        :
                else
                        cpu3=`echo "$cpu3" | cut -d "@" -f 1`
                fi;
                ##cpu4
                if [ $cpu4 == "off" ]; then
                        :
                else
                        cpu4=`echo "$cpu4" | cut -d "@" -f 1`
                fi;
		#percentage of cpu5
		if [ $cpu5 == "off" ]; then
                        :
                else
                        cpu5=`echo "$cpu5" | cut -d "@" -f 1`
                fi;
		if [ $cpu6 == "off" ]; then
                        :
                else
                        cpu6=`echo "$cpu6" | cut -d "@" -f 1`
                fi;
		#echo $a
		#separate the values of energy consumption for each vdd
		vdd11=`echo $vdd1 | cut -d "/" -f 1`
		vdd12=`echo $vdd1 | cut -d "/" -f 2`
		#the next 6 conditional deviations check which vdd value is greater from XXXX/YYYY form
		#first case: if the first sub-value is greater, than it is the chosen one
		if [ $vdd11 -gt $vdd12 ]; then
			#echo "vdd11 maior"
			#echo $vdd11
			vdd1=$vdd11;
		elif [ $vdd12 -gt $vdd11 ]; then
			#if the second value is greater, than it is chosen
			#echo "vdd12 maior"
                        vdd1=$vdd12;
		else
			#if both values are equal then the second value is chosen as default.
			#echo "iguais"
			vdd1=$vdd12;
		fi;
		#separate the values of energy consumption for each vdd
                vdd21=`echo $vdd2 | cut -d "/" -f 1`
                vdd22=`echo $vdd2 | cut -d "/" -f 2`
                if [ $vdd21 -gt $vdd22 ]; then
                        #echo "vdd21 maior"
                        vdd2=$vdd21;
                elif [ $vdd22 -gt $vdd21 ]; then
                        #echo "vdd22 maior"
                        vdd2=$vdd22;
                else
                        vdd2=$vdd22;
                fi;
		#separate the values of energy consumption for each vdd
                vdd31=`echo $vdd3 | cut -d "/" -f 1`
                vdd32=`echo $vdd3 | cut -d "/" -f 2`
                if [ $vdd31 -gt $vdd32 ]; then
                        #echo "vdd31 maior"
                        vdd3=$vdd31;
                elif [ $vdd32 -gt $vdd31 ]; then
                        #echo "vdd32 maior"
                        vdd3=$vdd32;
                else
                        vdd3=$vdd32;
                fi;
		#separate the values of energy consumption for each vdd
                vdd41=`echo $vdd4 | cut -d "/" -f 1`
                vdd42=`echo $vdd4 | cut -d "/" -f 2`
                if [ $vdd41 -gt $vdd42 ]; then
                        #echo "vdd41 maior"
                        vdd4=$vdd41;
                elif [ $vdd42 -gt $vdd41 ]; then
                        #echo "vdd42 maior"
                        vdd4=$vdd42;
                else
                        vdd4=$vdd42;
                fi;
		#separate the values of energy consumption for each vdd
                vdd51=`echo $vdd5 | cut -d "/" -f 1`
                vdd52=`echo $vdd5 | cut -d "/" -f 2`
                if [ $vdd51 -gt $vdd52 ]; then
                        #echo "vdd51 maior"
                        vdd5=$vdd51;
                elif [ $vdd52 -gt $vdd51 ]; then
                        #echo "vdd52 maior"
                        vdd5=$vdd52;
                else
                        vdd5=$vdd52;
                fi;
		#separate the values of energy consumption for each vdd
                vdd61=`echo $vdd6 | cut -d "/" -f 1`
                vdd62=`echo $vdd6 | cut -d "/" -f 2`
                if [ $vdd61 -gt $vdd62 ]; then
                        #echo "vdd51 maior"
                        vdd6=$vdd61;
                elif [ $vdd62 -gt $vdd61 ]; then
                        #echo "vdd62 maior"
                        vdd6=$vdd62;
                else
                        vdd6=$vdd62;
                fi;
		vddtotal=`echo "$vdd1 + $vdd2 + $vdd3 + $vdd4 + $vdd5 + $vdd6" | bc`;
		echo $ram","$cpu1","$cpu2","$cpu3","$cpu4","$cpu5","$cpu6","$gpu","$vdd1","$vdd2","$vdd3","$vdd4","$vdd5","$vdd6","$vddtotal
		#echo $f
		#echo $gpu
		#$vdd1 $vdd2 $vdd3 $vdd4 $vdd5 $vdd6
		done < "$i"
done
