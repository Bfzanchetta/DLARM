#!/bin/bash
arquivo=teste.txt
contador=0;
while IFS= read -r line
	do
	if [ $contador -eq 0 ]; then
		#corta o que quer e coloca na primeira posição
		epoch=`echo "$line" | cut -d "." -f 1`
		train=`echo "$line" | cut -d "," -f 2`
		test=`echo "$line" | cut -d "," -f 3`
		contador=$((contador+1));
		echo $epoch
		echo $train $test
	elif [ $contador -eq 1 ]; then
		#corta o que quer e coloca na segunda posição
		topone=`echo "$line" | cut -d " " -f 4`
		topfive=`echo "$line" | cut -d " " -f 5`
		contador=$((contador+1));
		echo $topone $topfive
	elif [ $contador -eq 2 ]; then
		#corta o que quer e coloca na terceira posição
		contador=$((contador+1));
	elif [ $contador -eq 3 ]; then
		#corta o que quer e coloca na quarta posição
		contador=0;
	else
		echo "Alerta Erro"
		:
	fi
	train=`echo "$line" | cut -d "," -f 2`
	tempo=`echo "$line" | cut -d "]" -f 2`
	#echo $train
	#echo $tempo
done < "$arquivo"
