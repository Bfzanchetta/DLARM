#!/bin/bash
path=/home/gppd/Área\ de\ Trabalho/folder
contador=0;
epochnumber=0;
trainacc=0;
testacc=0;
toponeerror=0;
topfiveerror=0;
tempo=0;
for i in `ls /home/gppd/Área\ de\ Trabalho/folder/*.txt`; do
	echo "epoch,train_acc,test_acc,toponeerror,topfiveerror,time" >> saida.csv
	while IFS= read -r line
		do
		if [ $contador -eq 0 ]; then
			#corta o que quer e coloca na primeira posição
			epoch=`echo "$line" | cut -d "." -f 1`
			epochnumber=`echo "$epoch" | cut -d " " -f 2`
			train=`echo "$line" | cut -d "," -f 2`
			test=`echo "$line" | cut -d "," -f 3`
			contador=$((contador+1));
			trainacc=`echo "$train" | cut -d " " -f 3`
			testacc=`echo "$test" | cut -d " " -f 3`
			#echo $epochnumber','$trainacc','$testacc >> saida.txt
		elif [ $contador -eq 1 ]; then
			#corta o que quer e coloca na segunda posição
			topone=`echo "$line" | cut -d " " -f 4`
			topfive=`echo "$line" | cut -d " " -f 5`
			contador=$((contador+1));
			#echo $topone $topfive
			toponeerror=`echo "$topone" | cut -d "=" -f 2`
			topfiveerror=`echo "$topfive" | cut -d "=" -f 2`
			echo "top5e1" $toponeerror $topfiveerror
		elif [ $contador -eq 2 ]; then
			tempo=`echo "$line" | cut -d ":" -f 2`
			tempo=`echo "$tempo" | cut -d " " -f 2`
			echo $epochnumber','$trainacc','$testacc','$toponeerror','$topfiveerror','$tempo >> saida.csv
			#corta o que quer e coloca na terceira posição
			contador=0;
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
done
